import tensorflow as tf

from functools import partial
from tfrecord_decoder import TfExampleDecoder


class Parser(object):

    def __init__(self,
                 output_size,
                 anchor_instance,
                 mode,
                 match_threshold=0.5,
                 unmatched_threshold=0.4,
                 number_of_max_fix_padding=100,
                 prototype_output_size=[138, 138],
                 skip_crowd_during_training=True,
                 use_bfloat16=True):

        self._mode = mode
        self._skip_crowd_during_training = skip_crowd_during_training
        self._is_training = (mode == "train")

        self._example_decoder = TfExampleDecoder()

        self._output_height = output_size[0]
        self._output_width = output_size[1]
        self._anchor_instance = anchor_instance
        self._match_threshold = match_threshold
        self._unmatched_threshold = unmatched_threshold

        self._number_of_max_fix_padding = number_of_max_fix_padding
        self.prototype_output_size = prototype_output_size

        self._use_bfloat16 = use_bfloat16
        self.count = 0

        if mode == "train":
            self._parse_fn = partial(self._parse, augment=True)
        elif mode == "val":
            self._parse_fn = partial(self._parse, augment=False)
        elif mode == "test":
            self._parse_fn = self._parse_predict_data

    def __call__(self, value):
        with tf.name_scope("parser"):
            data = self._example_decoder.decode(value)
            return self._parse_fn(data)

    def _parse(self, data, augment):
        classes = data['gt_classes']
        bboxes = data['gt_bboxes']
        masks = data['gt_masks']

        image = data['image']

        masks = tf.cast(masks, tf.bool)
        masks = tf.cast(masks, tf.float32)

        masks = tf.cast(masks + 0.5, tf.uint8)

        masks = tf.expand_dims(masks, axis=-1)

        image = tf.image.resize(
            image, [self._output_height, self._output_width])
        masks = tf.image.resize(masks,
                                [self.prototype_output_size[0],
                                    self.prototype_output_size[1]],
                                method=tf.image.ResizeMethod.BILINEAR)

        masks = tf.squeeze(masks)
        masks = tf.cast(masks + 0.5, tf.uint8)
        masks = tf.cast(masks, tf.float32)

        normalized_bboxes = bboxes
        bboxes = bboxes * [self._output_height, self._output_width,
                           self._output_height, self._output_width]

        all_offsets, conf_gt, prior_max_box, prior_max_index = self._anchor_instance.matching(
            self._match_threshold, self._unmatched_threshold, bboxes, classes)

        number_of_objects = tf.size(classes)

        number_of_paddings = self._number_of_max_fix_padding - \
            tf.shape(classes)[0]
        padded_classes = tf.zeros([number_of_paddings], dtype=tf.int64)
        padded_bboxes = tf.zeros([number_of_paddings, 4])
        padded_masks = tf.zeros(
            [number_of_paddings,
             self.prototype_output_size[0],
             self.prototype_output_size[1]])
        normalized_bboxes = tf.concat(
            [normalized_bboxes, padded_bboxes], axis=0)

        if tf.shape(classes)[0] == 1:
            masks = tf.expand_dims(masks, axis=0)

        masks = tf.concat([masks, padded_masks], axis=0)
        classes = tf.concat([classes, padded_classes], axis=0)

        labels = {
            'all_offsets': all_offsets,
            'conf_gt': conf_gt,
            'prior_max_box': prior_max_box,
            'prior_max_index': prior_max_index,
            'normalized_boxes': normalized_bboxes,
            'classes': classes,
            'number_of_objects': number_of_objects,
            'target_masks': masks
        }

        return (image, labels)
