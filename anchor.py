from itertools import product
from math import sqrt

import tensorflow as tf


class Anchor(object):
    def __init__(self, image_height, image_width, feature_map_sizes, aspect_ratios, scales):
        self.image_height = image_height
        self.image_width = image_width
        self.number_of_anchors, self.anchors_normal = self._generate_anchors(
            feature_map_sizes, aspect_ratios, scales)
        self.anchors = self.get_anchors()

    def _generate_anchors(self, feature_map_sizes, aspect_ratios, scales):
        prior_boxes = []
        number_of_anchors = 0

        for index, feature_map_size in enumerate(feature_map_sizes):
            anchor_count = 0
            for j, i in product(range(int(feature_map_size[0])), range(int(feature_map_size[1]))):
                x = (i + 0.5) / feature_map_size[1]
                y = (j + 0.5) / feature_map_size[0]
                for aspect_ratio in aspect_ratios:
                    a = sqrt(aspect_ratio)
                    w = scales[index] * a / feature_map_size[1]
                    h = scales[index] / a / feature_map_size[0]

                    prior_boxes += [x, y, w, h]
                    anchor_count += 1
            number_of_anchors += anchor_count
            print(f'Create priors for featuremap size: {feature_map_size} ',
                  f'aspect ratio: {aspect_ratio} ',
                  f'scale: {scales[index]} ',
                  f'number of anchors: {anchor_count}')

            output = tf.reshape(tf.convert_to_tensor(prior_boxes), [-1, 4])
            output = tf.cast(output, tf.float32)

            return number_of_anchors, output

    def _encode(self, map_location, anchors, include_variances=False):
        gh = map_location[:, 2] - map_location[:, 0]
        gw = map_location[:, 3] - map_location[:, 1]
        center_gt = tf.cast(tf.stack(
            [map_location[:, 1] + (gw / 2),
             map_location[:, 0] + (gh / 2), gw, gh],
            axis=-1), tf.float32)

        ph = anchors[:, 2] - anchors[:, 0]
        pw = anchors[:, 3] - anchors[:, 1]
        center_anchors = tf.cast(tf.stack(
            [anchors[:, 1] + (pw / 2),
             anchors[:, 0] + (ph / 2), pw, ph],
            axis=-1), tf.float32)
        variances = [0.1, 0.2]

        if include_variances:
            g_hat_cx = (center_gt[:, 0] - center_anchors[:, 0]
                        ) / center_anchors[:, 2] / variances[0]
            g_hat_cy = (center_gt[:, 1] - center_anchors[:, 1]
                        ) / center_anchors[:, 3] / variances[0]
        else:
            g_hat_cx = (center_gt[:, 0] - center_anchors[:, 0]
                        ) / center_anchors[:, 2]
            g_hat_cy = (center_gt[:, 1] - center_anchors[:, 1]
                        ) / center_anchors[:, 3]

        tf.debugging.assert_non_negative(
            center_anchors[:, 2] / center_gt[:, 2])
        tf.debugging.assert_non_negative(
            center_anchors[:, 3] / center_gt[:, 3])

        if include_variances:
            g_hat_w = tf.math.log(center_gt[:, 2] / center_anchors[:, 2]
                                  ) / variances[1]
            g_hat_h = tf.math.log(center_gt[:, 3] / center_anchors[:, 3]
                                  ) / variances[1]
        else:
            g_hat_w = tf.math.log(center_gt[:, 2] / center_anchors[:, 2])
            g_hat_h = tf.math.log(center_gt[:, 3] / center_anchors[:, 3])

        tf.debugging.assert_all_finite(g_hat_w,
                                       "Ground truth box width encoding is NaN/Infinite")
        tf.debugging.assert_all_finite(g_hat_h,
                                       "Ground truth box height encoding is NaN/Infinite")
        offsets = tf.stack([g_hat_cx, g_hat_cy, g_hat_w, g_hat_h], axis=-1)
        return offsets

    def _area(self, boxlist, scope=None):
        y_min, x_min, y_max, x_max, = tf.split(
            value=boxlist, num_or_size_splits=4, axis=1)
        return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])

    def _intersection(self, boxlist1, boxlist2):
        y_min1, x_min1, y_max1, x_max1 = tf.split(
            value=boxlist1, num_or_size_splits=4, axis=1)
        y_min2, x_min2, y_max2, x_max2 = tf.split(
            value=boxlist2, num_or_size_splits=4, axis=1)

        all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
        all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))

        intersect_heights = tf.maximum(
            0.0, all_pairs_min_ymax - all_pairs_max_ymin)
        all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
        all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
        intersect_widths = tf.maximum(
            0.0, all_pairs_min_xmax - all_pairs_max_xmin)

        return intersect_heights * intersect_widths

    def _iou(self, boxlist1, boxlist2):
        intersections = self._intersection(boxlist1, boxlist2)
        areas1 = self._area(boxlist1)
        areas2 = self._area(boxlist2)
        unions = (tf.expand_dims(areas1, 1) +
                  tf.expand_dims(areas2, 0) - intersections)

        return tf.where(
            tf.equal(intersections, 0.0),
            tf.zeros_like(intersections),
            tf.truediv(intersections, unions))

    def get_anchors(self):
        w = self.anchors_normal[:, 2]
        h = self.anchors_normal[:, 3]

        anchors_yxyx = tf.cast(tf.stack(
            [
                (self.anchors_normal[:, 1] - (h / 2)) * self.image_height,
                (self.anchors_normal[:, 0] - (w / 2)) * self.image_width,
                (self.anchors_normal[:, 1] - (h / 2)) * self.image_height,
                (self.anchors_normal[:, 0] - (w / 2)) * self.image_width
            ], axis=-1), tf.float32)

        return anchors_yxyx

    def matching(self,
                 positive_threshold,
                 negative_threshold,
                 ground_truth_bboxes,
                 ground_truth_labels):
        pairwise_iou = self._iou(self.anchors, ground_truth_bboxes)

        each_prior_max = tf.reduce_max(pairwise_iou, axis=-1)

        if tf.shape(pairwise_iou)[-1] == 0:
            return (self.anchors * 0, tf.cast(self.anchors[:, 0] * 0, dtype=tf.int64),
                self.anchors * 0, tf.cast(self.anchors[:, 0] * 0, dtype=tf.int64))

        each_prior_index = tf.math.argmax(pairwise_iou, axis=-1)

        each_box_index = tf.math.argmax(pairwise_iou, axis=0)

        indices = tf.expand_dims(each_box_index, axis=-1)

        updates = tf.cast(tf.tile(tf.constant([2]), tf.shape(each_box_index)), dtype=tf.float32)
        each_prior_max = tf.tensor_scatter_nd_update(each_prior_max, indices, updates)

        updates = tf.cast(tf.range(0, tf.shape(each_box_index)[0]), dtype=tf.int64)
        each_prior_index = tf.tensor_scatter_nd_update(each_prior_index, indices, updates)

        each_prior_box = tf.gather(ground_truth_bboxes, each_prior_index)

        conf = tf.squeeze(tf.gather(ground_truth_labels, each_prior_index))

        neutral_label_index = tf.where(each_prior_max < positive_threshold)
        background_label_index = tf.where(each_prior_max < negative_threshold)

        conf = tf.tensor_scatter_nd_update(conf, neutral_label_index, -1 * tf.ones(tf.size(neutral_label_index), dtype=tf.int64))
        conf = tf.tensor_scatter_nd_update(conf, background_label_index, tf.zeros(tf.size(background_label_index), dtype=tf.int64))

        offsets = self._encode(each_prior_box, self.anchors)

        return offsets, conf, each_prior_box, each_prior_index
