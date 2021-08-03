import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

from layers.fpn import FPN
from layers.prediction import PredictionModule
from layers.protonet import ProtoNet
from detection import Detect
import anchor


class Yolact(tf.keras.Model):

    def __init__(self, image_height, image_width, fpn_channels, number_of_classes,
                 number_of_masks, aspect_ratios, scales, base_model_trainable=False):
        super(Yolact, self).__init__()
        out = ['conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']

        base_model = tf.keras.applications.resnet50.ResNet50(
            input_shape=(image_height, image_width, 3),
            include_top=False,
            layers=tf.keras.layers,
            weights='imagenet'
        )

        base_model.trainable = base_model_trainable

        self.backbone_resnet = tf.keras.Model(inputs=base_model.input,
                                              outputs=[base_model.get_layer(x).output
                                                       for x in out])
        self.feature_map_size = np.array(
            [list(base_model.get_layer(x).output.shape[1:3]) for x in out])

        out_height_p6 = np.ceil(
            (self.feature_map_size[-1, 0]).astype(np.float32) / 2.0)
        out_width_p6 = np.ceil(
            (self.feature_map_size[-1, 1]).astype(np.float32) / 2.0)
        out_height_p7 = np.ceil(out_height_p6 / 2.0)
        out_width_p7 = np.ceil(out_width_p6 / 2.0)

        self.feature_map_size = np.concatenate(
            (self.feature_map_size, [
             [out_height_p6, out_width_p6], [out_height_p7, out_width_p7]]),
            axis=0)

        self.protonet_out_size = self.feature_map_size[0] * 2.0

        self.backbone_fpn = FPN(fpn_channels)
        self.protonet = ProtoNet(number_of_masks)

        self.semantic_segmentation = tf.keras.layers.Conv2D(
            number_of_classes - 1, (1, 1), 1, padding='same',
            kernel_initializer=tf.keras.initializers.glorot_uniform())

        anchorobject = anchor.Anchor(
            image_height=image_height,
            image_width=image_width,
            feature_map_sizes=self.feature_map_size,
            aspect_ratios=aspect_ratios,
            scales=scales
        )

        self.number_of_anchors = anchorobject.number_of_anchors
        self.priors = anchorobject.anchors

        self.predictionHead = PredictionModule(256, len(aspect_ratios),
                                               number_of_classes, number_of_masks)

        self.detect = Detect(number_of_classes, conf_threshold=0.05, nms_threshold=0.5)
        self.max_output_size = 300

    @tf.function
    def call(self, inputs, training=False):
        inputs = tf.cast(inputs, tf.float32)
        inputs = tf.keras.applications.resnet50.preprocess_input(inputs)

        c3, c4, c5 = self.backbone_resnet(inputs, training=False)
        fpn_out = self.backbone_fpn(c3, c4, c5)

        p3 = fpn_out[0]
        protonet_out = self.protonet(p3)

        segmentation = self.semantic_segmentation(p3)

        predicted_classes = []
        predicted_offsets = []
        predicted_mask_coefficients = []

        for feature_map in fpn_out:
            _class, offset, coefficient = self.predictionHead(feature_map)
            predicted_classes.append(_class)
            predicted_offsets.append(offset)
            predicted_mask_coefficients.append(coefficient)

        predicted_classes = tf.concat(predicted_classes, axis=1)
        predicted_offsets = tf.concat(predicted_offsets, axis=1)
        predicted_mask_coefficients = tf.concat(predicted_mask_coefficients, axis=1)

        if training:
            pred = {
                'predicted_classes': predicted_classes,
                'predicted_offsets': predicted_offsets,
                'predicted_mask_coefficients': predicted_mask_coefficients,
                'prototype_out': protonet_out,
                'segmentation': segmentation,
                'priors': self.priors
            }

            result = {
                'detection_boxes': tf.zeros((self.max_output_size, 4)),
                'detection_classes': tf.zeros((self.max_output_size)),
                'detection_scores': tf.zeros((self.max_output_size)),
                'detection_masks': tf.zeros((self.max_output_size, 30, 30, 1)),
                'number_of_detections': tf.constant(0)
            }

            pred.update(result)
        else:
            pred = {
                'predicted_classes': predicted_classes,
                'predicted_offsets': predicted_offsets,
                'predicted_mask_coefficients': predicted_mask_coefficients,
                'prototype_out': protonet_out,
                'segmentation': segmentation,
                'priors': self.priors
            }

            pred.update(self.detect(pred, img_shape=tf.shape(inputs)))

        return pred
