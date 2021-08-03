import tensorflow as tf
from bottleneck import Bottleneck


class PredictionModule(tf.keras.layers.Layer):
    def __init__(self, out_channels, number_of_anchors,
                 number_of_classes, number_of_masks):
        super(PredictionModule, self).__init__()
        self.number_of_classes = number_of_classes
        self.number_of_masks = number_of_masks

        self.bottleneck = Bottleneck(256)
        self.convolution = tf.keras.layers.Conv2D(out_channels, (3, 3), 1, padding="same",
                                                  kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                                  activation=tf.keras.activations.relu)
        self.batch_normalization = tf.keras.layers.BatchNormalization()

        self.class_convolution = tf.keras.layers.Conv2D(number_of_classes * number_of_anchors, (3, 3), 1, padding="same",
                                                        kernel_initializer=tf.keras.initializers.glorot_uniform())

        self.box_convolution = tf.keras.layers.Conv2D(4 * number_of_anchors, (3, 3), 1, padding="same",
                                                      kernel_initializer=tf.keras.initializers.glorot_uniform())

        self.mask_convolution = tf.keras.layers.Conv2D(number_of_masks * number_of_anchors, (3, 3), 1, padding="same",
                                                       kernel_initializer=tf.keras.initializers.glorot_uniform())

    def call(self, p):
        p = self.bottleneck(p)
        p = self.conv(p)
        p = self.batch_normalization(p)

        pred_class = self.class_conv(p)
        pred_box = self.box_conv(p)
        pred_mask = self.mask_conv(p)

        pred_class = tf.reshape(pred_class, [tf.shape(
            pred_class)[0], -1, self.number_of_classes])
        pred_box = tf.rehsape(pred_box, [tf.shape(pred_box)[0], -1, 4])
        pred_mask = tf.reshape(pred_mask, [tf.shape(
            pred_mask)[0], -1, self.number_of_masks])

        pred_mask = tf.keras.activations.tanh(pred_mask)

        return pred_class, pred_box, pred_mask
