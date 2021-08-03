import tensorflow as tf


class ProtoNet(tf.keras.layers.Layer):
    def __init__(self, number_of_prototypes):
        super(ProtoNet, self).__init__()
        self.convolution = [
            tf.keras.layers.Conv2D(256, (3, 3), 1, padding="same",
                                   kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                   activation=tf.keras.activations.relu)
            for _ in range(4)
        ]

        self.up_sampling = tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation="bilinear")
        self.final_convolution = tf.keras.layers.Conv2D(number_of_prototypes, (1, 1), 1, padding="same",
                                                        kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                                        activation=tf.keras.activations.relu)

    def call(self, p3):
        proto = self.convolution[0](p3)
        proto = self.convolution[1](proto)
        proto = self.convolution[2](proto)

        proto = tf.keras.activations.relu(self.up_sampling(proto))
        proto = self.convolution[3](proto)

        proto = self.final_convolution(proto)
        return proto
