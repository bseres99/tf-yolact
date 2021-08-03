import tensorflow as tf


class FPN(tf.keras.layers.Layer):
    def __init__(self, number_of_filters):
        super(FPN, self).__init__()

        self.up_sample = tf.keras.layers.UpSampling2D(interpolation="bilinear")

        self.down_sample = [
            tf.keras.layers.Conv2D(number_of_filters, (3, 3), padding="same",
                                   kernel_initializer=tf.keras.initializers.glorot_uniform()),
            tf.keras.layers.Conv2D(number_of_filters, (3, 3), padding="same",
                                   kernel_initializer=tf.keras.initializers.glorot_uniform())
        ]

        self.lateral_convolution = [
            tf.keras.layers.Conv2D(number_of_filters, (1, 1), padding="same",
                                   kernel_initializer=tf.keras.initializers.glorot_uniform()),
            tf.keras.layers.Conv2D(number_of_filters, (1, 1), padding="same",
                                   kernel_initializer=tf.keras.initializers.glorot_uniform()),
            tf.keras.layers.Conv2D(number_of_filters, (1, 1), padding="same",
                                   kernel_initializer=tf.keras.initializers.glorot_uniform())
        ]

        self.predict = [
            tf.keras.layers.Conv2D(number_of_filters, (3, 3), padding="same",
                                   kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                   activation=tf.keras.activations.relu),
            tf.keras.layers.Conv2D(number_of_filters, (3, 3), padding="same",
                                   kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                   activation=tf.keras.activations.relu),
            tf.keras.layers.Conv2D(number_of_filters, (3, 3), padding="same",
                                   kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                   activation=tf.keras.activations.relu)
        ]

    def call(self, c3, c4, c5):
        p5 = self.lateral_convolution[0](c5)
        p4 = self.crop_and_add(self.up_sample(
            p5), self.lateral_convolution[1](c4))
        p3 = self.crop_and_add(self.up_sample(
            p4), self.lateral_convolution[2](c3))
        print(f"p3: {p3.shape}")

        p3 = self.predict[0](p3)
        p4 = self.predict[1](p4)
        p5 = self.predict[2](p5)

        p6 = self.down_sample[0](p5)
        p7 = self.down_sample[1](p6)

        return [p3, p4, p5, p6, p7]

    def crop_and_add(self, x1, x2):
        offsets = [0, (x1.shape[1] - x2.shape[1]) // 2,
                   (x1.shape[2] - x2.shape[2]) // 2, 0]
        size = [-1, x2.shape[1], x2.shape[2], -1]

        return tf.add(tf.slice(x1, offsets, size), x2)
