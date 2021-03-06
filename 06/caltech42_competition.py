#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub  # Note: you need to install tensorflow_hub

from caltech42 import Caltech42

# The neural network model


class Network:
    def __init__(self, args):

        # TODO: You should define `self.model`. You should use the following layer:
        #   mobilenet = tfhub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2", output_shape=[1280])
        # The layer:
        # - if given `trainable=True/False` to KerasLayer constructor, the layer weights
        #   either are marked or not marked as updatable by an optimizer;
        # - however, batch normalization regime is set independently, by `training=True/False`
        #   passed during layer execution.
        #
        # Therefore, to not train the layer at all, you should use
        #   mobilenet = tfhub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2", output_shape=[1280], trainable=False)
        #   features = mobilenet(inputs, training=False)
        # On the other hand, to fully train it, you should use
        #   mobilenet = tfhub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2", output_shape=[1280], trainable=True)
        #   features = mobilenet(inputs)
        # where the `training` argument to `mobilenet` is passed automatically in that case.
        #
        # Note that a model with KerasLayer can currently be saved only using
        #   tf.keras.experimental.export_saved_model(model, path, serving_only=True/False)
        # where `serving_only` controls whether only prediction, or also training/evaluation
        # graphs are saved. To again load the model, use
        #   model = tf.keras.experimental.load_from_saved_model(path, {"KerasLayer": tfhub.KerasLayer})

        input_shape = (Caltech42.MIN_SIZE, Caltech42.MIN_SIZE, Caltech42.C)

        mobilnet = tfhub.KerasLayer(
            "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2", input_shape=input_shape, output_shape=[1280], trainable=False)

        inputs = tf.keras.layers.Input(shape=input_shape)
        features = mobilnet(inputs, training=False)
        features = tf.keras.layers.Dense(1024, activation='relu')(features)
        features = tf.keras.layers.Dropout(0.2)(features)
        features = tf.keras.layers.Dense(512, activation='relu')(features)
        features = tf.keras.layers.Dropout(0.2)(features)
        #features = tf.keras.layers.Dense(512, activation='relu')(features)
        #features = tf.keras.layers.Dense(256, activation='relu')(features)
        outputs = tf.keras.layers.Dense(
            Caltech42.LABELS, activation=tf.nn.softmax)(features)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy")],
        )

        self.tb_callback = tf.keras.callbacks.TensorBoard(
            args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None

    def train(self, caltech42, args):

        self.model.fit(
            caltech42.train.data["images"], caltech42.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(
                caltech42.dev.data["images"], caltech42.dev.data["labels"]),
            callbacks=[self.tb_callback],
        )
        pass

    def predict(self, caltech42, args):
        test_logs = self.model.predict(
            caltech42.data["images"])
        return test_logs


def processImage(image):
    t = tf.io.decode_image(image, channels=3, dtype=tf.dtypes.float32)
    resized = tf.image.resize_image_with_crop_or_pad(t, 224, 224)
    nparr = resized.numpy()
    return nparr


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=40,
                        type=int, help="Batch size.")
    parser.add_argument("--epochs", default=30,
                        type=int, help="Number of epochs.")
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
    caltech42 = Caltech42(processImage)

    # Create the network and train
    network = Network(args)
    network.train(caltech42, args)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "caltech42_competition_test.txt"), "w", encoding="utf-8") as out_file:
        for probs in network.predict(caltech42.test, args):
            print(np.argmax(probs), file=out_file)
