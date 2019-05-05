# Jan Pavlovsky

#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from mnist import MNIST

# The neural network model


class Network(tf.keras.Model):
    def __init__(self, args):
        inputs = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])

        # TODO: Add CNN layers specified by `args.cnn`, which contains
        # comma-separated list of the following layers:
        # - `C-filters-kernel_size-stride-padding`: Add a convolutional layer with ReLU
        #   activation and specified number of filters, kernel size, stride and padding.
        # - `CB-filters-kernel_size-stride-padding`: Same as `C`, but use batch normalization.
        #   In detail, start with a convolutional layer without bias and activation,
        #   then add batch normalization layer, and finally ReLU activation.
        # - `M-kernel_size-stride`: Add max pooling with specified size and stride.
        # - `R-[layers]`: Add a residual connection. The `layers` contain a specification
        #   of at least one convolutional layer (but not a recursive residual connection `R`).
        #   The input to the specified layers is then added to their output.
        # - `F`: Flatten inputs. Must appear exactly once in the architecture.
        # - `D-hidden_layer_size`: Add a dense layer with ReLU activation and specified size.
        # Produce the results in variable `hidden`.

        # Add the final output layer

        layers = [x.strip() for x in args.cnn.split(',')]

        layerIndex = 0
        inResidualBlock = False

        self.current = inputs

        for layer in layers:
            tokens = layer.split('-')

            if(tokens[0] == 'R' or inResidualBlock):
                if(tokens[0] == 'R'):
                    self.shortcut = self.current

                if(tokens[4][-1:] == ']'):
                    self.current = self.EndResBlock(
                        self.shortcut, layerIndex, tokens)
                    layerIndex += 1
                    inResidualBlock = False
                else:
                    inResidualBlock = True
                    param = tokens[2:]
                    self.current = self.ConvLayer(*param)

            elif(tokens[0] == 'F'):
                self.current = self.FlattenLayer()
            elif(tokens[0] == 'C'):
                param = tokens[1:]
                self.current = self.ConvLayer(*param)
            elif(tokens[0] == 'CB'):
                param = tokens[1:]
                self.current = self.ConvBatchNorm(*param)
            elif(tokens[0] == 'M'):
                param = tokens[1:]
                self.current = self.MaxPooling(*param)
            elif(tokens[0] == 'D'):
                self.current = self.DenseLayer(tokens[1])

            layerIndex += 1

        hidden = self.current

        outputs = tf.keras.layers.Dense(
            MNIST.LABELS, activation=tf.nn.softmax)(hidden)

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
        return tf.keras.layers.Dense(int(kernel_size), activation='relu')(self.current)

    def EndResBlock(self, shortcut, layerIndex, tokens):
        tokens[4] = tokens[4][:-1]
        param = tokens[1:]
        self.current = self.ConvLayer(*param)
        return tf.keras.layers.Add()([self.shortcut, self.current])

    def train(self, mnist, args):
        self.fit(
            mnist.train.data["images"], mnist.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(
                mnist.dev.data["images"], mnist.dev.data["labels"]),
            callbacks=[self.tb_callback],
        )

    def test(self, mnist, args):
        test_logs = self.evaluate(
            mnist.test.data["images"], mnist.test.data["labels"], batch_size=args.batch_size)
        self.tb_callback.on_epoch_end(1, dict(
            ("val_test_" + metric, value) for metric, value in zip(self.metrics_names, test_logs)))
        return test_logs[self.metrics_names.index("accuracy")]


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50,
                        type=int, help="Batch size.")
    parser.add_argument("--cnn", default="C-8-3-5-valid,R-[C-8-3-1-same,C-8-3-1-same],F,D-50",
                        type=str, help="CNN architecture.")
    parser.add_argument("--epochs", default=5, type=int,
                        help="Number of epochs.")
    parser.add_argument("--recodex", default=False,
                        action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--threads", default=1, type=int,
                        help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    if args.recodex:
        tf.keras.utils.get_custom_objects(
        )["glorot_uniform"] = lambda: tf.keras.initializers.glorot_uniform(seed=42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub(
            "(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    mnist = MNIST()

    # Create the network and train
    network = Network(args)
    network.train(mnist, args)

    # Compute test set accuracy and print it
    accuracy = network.test(mnist, args)
    with open("mnist_cnn.out", "w") as out_file:
        print("{:.2f}".format(100 * accuracy), file=out_file)
