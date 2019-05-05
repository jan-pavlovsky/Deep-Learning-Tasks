# Jan Pavlovsky

#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from cifar10 import CIFAR10

# The neural network model


class Network(tf.keras.Model):
    def __init__(self, args):
        # TODO: Define a suitable model, by calling `super().__init__`
        # with appropriate inputs and outputs.
        #
        # Alternatively, if you prefer to use a `tf.keras.Sequential`,
        # replace the `Network` parent, call `super().__init__` at the beginning
        # of this constructor and add layers using `self.add`.

        # TODO: After creating the model, call `self.compile` with appropriate arguments.

        inputs = tf.keras.layers.Input(shape=[32, 32, 3])
        self.current = inputs

        self.current = self.ConvBatchNorm(32, 5, 1, 'same')
        self.current = self.ConvBatchNorm(64, 3, 1, 'same')
        self.current = self.MaxPooling(2, 2)
        self.current = tf.keras.layers.Dropout(0.3)(self.current)
        self.current = self.ConvBatchNorm(64, 3, 1, 'same')
        self.current = self.ConvBatchNorm(128, 3, 1, 'same')
        self.current = self.MaxPooling(2, 2)
        self.current = self.FlattenLayer()
        #self.current = self.DenseLayer(512)
        #self.current = self.DenseLayer(256)

        outputs = tf.keras.layers.Dense(
            10, activation=tf.nn.softmax)(self.current)

        super().__init__(inputs=inputs, outputs=outputs)

        self.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy")],
        )

        self.tb_callback = tf.keras.callbacks.TensorBoard(
            args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None

    def train(self, cifar, args):
        self.fit(
            cifar.train.data["images"], cifar.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(
                cifar.dev.data["images"], cifar.dev.data["labels"]),
            callbacks=[self.tb_callback],
        )

    def FlattenLayer(self):
        return tf.keras.layers.Flatten()(self.current)

    def ConvLayer(self, filters, kernel, stride, padding):
        return tf.keras.layers.Conv2D(int(filters), (int(kernel), int(kernel)), strides=(
            int(stride), int(stride)), padding=padding, activation='relu')(self.current)

    def ConvBatchNorm(self, filters, kernel, stride, padding):
        self.current = tf.keras.layers.Conv2D(int(filters), (int(kernel), int(kernel)), strides=(
            int(stride), int(stride)), padding=padding, activation=None, use_bias=False)(self.current)
        self.current = tf.keras.layers.BatchNormalization()(self.current)
        return tf.keras.layers.Activation("relu")(self.current)

    def MaxPooling(self, kernel, stride="1"):
        return tf.keras.layers.MaxPooling2D(
            pool_size=(int(kernel), int(kernel)), strides=(int(stride), int(stride)))(self.current)

    def DenseLayer(self, kernel_size):
        dense = tf.keras.layers.Dense(
            int(kernel_size), activation='relu')(self.current)
        return tf.keras.layers.Dropout(0.2)(dense)

    def EndResBlock(self):
        return tf.keras.layers.Add()([self.shortcut, self.current])

    def StartResBlock(self):
        self.shortcut = self.current
        return self.current


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=100,
                        type=int, help="Batch size.")
    parser.add_argument("--epochs", default=8, type=int,
                        help="Number of epochs.")
    parser.add_argument("--threads", default=4, type=int,
                        help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub(
            "(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load data
    cifar = CIFAR10()

    # Create the network and train
    network = Network(args)
    network.train(cifar, args)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as out_file:
        for probs in network.predict(cifar.test.data["images"], batch_size=args.batch_size):
            print(np.argmax(probs), file=out_file)
