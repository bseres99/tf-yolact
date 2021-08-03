import tensorflow as tf


class Bottleneck(tf.keras.layers.Layer):
    def __init__(self, planes, stride=1, normalization_layer=tf.keras.layers.BatchNormalization(), dilation=1):
        super(Bottleneck, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(
            planes, (1, 1), dilation_rate=(dilation, dilation))
        self.bn1 = normalization_layer

        self.conv2 = tf.keras.layers.Conv2D(planes, (3, 3), strides=(
            stride, stride), dilation_rate=(dilation, dilation))
        self.bn2 = normalization_layer
        self.conv3 = tf.keras.layers.Conv2D(
            planes, kernel_size=(1, 1), dilation_rate=(dilation, dilation))
        self.bn3 = normalization_layer

    def call(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = tf.nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = tf.nn.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = tf.nn.relu(out)

        return out
