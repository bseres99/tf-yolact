import numpy as np
import tensorflow as tf


class Detect():
    def __init__(self, number_of_classes, conf_threshold,
                 nms_threshold, max_output_size=300, per_class_max_output_size=100):

        assert(nms_threshold <= 0, "nms_threshold must be non negative.")
        self.nms_threshold = nms_threshold
        self.number_of_classes = number_of_classes
        self.conf_threshold = conf_threshold
        self.use_fast_nms = False
        self.max_output_size = max_output_size
        self.per_class_max_output_size = per_class_max_output_size

    def __call__(self, net_outs, img_shape: list[int], use_cropped_mask=True):
        bounding_box_predictions = net_outs["pred_offset"]
        class_predictions = net_outs["pred_cls"]
        coefficient_predictions = net_outs["pred_mask_coef"]
        anchors = net_outs["priors"]
        prototype_predictions = net_outs["proto_out"]

        prototype_height = tf.shape(prototype_predictions)[1]
        prototype_width = tf.shape(prototype_predictions)[2]

        class_predictions = tf.nn.softmax(class_predictions, axis=-1)
        class_predictions = class_predictions[:, :, 1:]

        class_prediction_max = tf.reduce_max(class_predictions, axis=-1)
        batch_size = tf.shape(class_prediction_max)[0]

        detection_boxes = tf.zeros(
            (batch_size, self.max_output_size, 4), tf.float32)
        detection_classes = tf.zeros(
            (batch_size, self.max_output_size), tf.float32)
        detection_scores = tf.zeros(
            (batch_size, self.max_output_size), tf.float32)
        detection_masks = tf.zeros(
            (batch_size, self.max_output_size, prototype_height, prototype_width), tf.float32)
        number_of_detections = tf.zeros((batch_size), tf.int32)

        for b in range(batch_size):
            class_threshold = tf.boolean_mask(
                class_predictions[b], class_prediction_max[b] > self.conf_threshold)
            coefficient_threshold = tf.boolean_mask(
                coefficient_predictions[b], class_prediction_max[b] > self.conf_threshold)
            box_threshold = tf.boolean_mask(
                bounding_box_predictions[b], class_prediction_max[b] > self.conf_threshold)
            anchor_threshold = tf.boolean_mask(
                anchors[b], class_prediction_max[b] > self.conf_threshold)

            decoded_bounding_box = self._decode(
                box_threshold, anchor_threshold)
            if tf.size(class_threshold) != 0:
                box_threshold, coefficient_threshold, class_ids, class_threshold = self._traditional_nms(
                    box_threshold, coefficient_threshold, class_threshold)

                number_of_detections = tf.shape(box_threshold)[0]

                masks = tf.matmul(
                    prototype_predictions[b], tf.transpose(coefficient_threshold))
                masks = tf.sigmoid(masks)

                boxes = self._sanitize(
                    box_threshold, width=img_shape[2], height=img_shape[1])
                boxes = tf.stack([
                    boxes[:, 0] / tf.cast(img_shape[1], tf.float32),
                    boxes[:, 1] / tf.cast(img_shape[2], tf.float32),
                    boxes[:, 2] / tf.cast(img_shape[1], tf.float32),
                    boxes[:, 3] / tf.cast(img_shape[2], tf.float32)
                ], axis=-1)

                masks = tf.clip_by_value(
                    masks, clip_value_min=0.0, clip_value_max=1.0)
                masks = tf.transpose(masks, (2, 0, 1))

                _ind_boxes = tf.stack(
                    (tf.tile([b], number_of_detections), tf.range(0, tf.shape(boxes)[0])), axis=-1)
                detection_boxes = tf.tensor_scatter_nd_update(
                    detection_boxes, _ind_boxes, boxes)
                detection_classes = tf.tensor_scatter_nd_update(
                    detection_classes, _ind_boxes, class_ids)
                detection_scores = tf.tensor_scatter_nd_update(
                    detection_scores, _ind_boxes, class_threshold)
                detection_masks = tf.tensor_scatter_nd_update(
                    detection_masks, _ind_boxes, masks)
                number_of_detections = tf.tensor_scatter_nd_update(
                    number_of_detections, [[b]], number_of_detections)

        result = {"detection_boxes": detection_boxes, "detection_classes": detection_classes,
                  "detection_scores": detection_scores, "detection_masks": detection_masks, "number_of_detections": number_of_detections}
        return result

    def _decode(self, box_p, priors, include_variances=False):
        variances = [0.1, 0.2]
        box_p = tf.cast(box_p, tf.float32)
        priors = tf.cast(priors, tf.float32)

        ph = priors[:, 2] - priors[:, 0]
        pw = priors[:, 3] - priors[:, 1]

        priors = tf.cast(tf.stack(
            [priors[:, 1] + (pw / 2),
             priors[:, 0] + (ph / 2), pw, ph],
            axis=-1), tf.float32)

        if include_variances:
            b_x_y = priors[:, :2] + box_p[:, :, :2] * \
                priors[:, 2:] * variances[0]
            b_w_h = priors[:, 2:] * tf.math.exp(box_p[:, :, 2:] * variances[1])
        else:
            b_x_y = priors[:, :2] + box_p[:, :, :2] * priors[:, 2:]
            b_w_h = priors[:, 2:] * tf.math.exp(box_p[:, :, 2:])

        boxes = tf.concat([b_x_y, b_w_h], axis=-1)

        boxes = tf.concat([boxes[:, :, :2] - boxes[:, :, 2:] / 2,
                          boxes[:, :, 2:] / 2 + boxes[:, :, :2]], axis=-1)

        return tf.stack([boxes[:, :, 1], boxes[:, :, 0], boxes[:, :, 3], boxes[:, :, 2]], axis=-1)

    def _sanitize_coordinates(self, _x1, _x2, size, padding: int = 0):
        x1 = tf.math.minimum(_x1, _x2)
        x2 = tf.math.maximum(_x1, _x2)
        x1 = tf.clip_by_value(x1 - padding, clip_value_min=0.0,
                              clip_value_max=tf.cast(1.0, tf.float32))
        x2 = tf.clip_by_value(x2 + padding, clip_value_min=0.0,
                              clip_value_max=tf.cast(1.0, tf.float32))

        return x1, x2

    def _sanitize(self, boxes, width, height, padding: int = 0, crop_size=(30, 30)):
        x1, x2 = self._sanitize_coordinates(
            boxes[:, 1], boxes[:, 3], width, padding)
        y1, y2 = self._sanitize_coordinates(
            boxes[:, 0], boxes[:, 2], height, padding)

        boxes = tf.stack((y1, x1, y2, x2), axis=1)

        return boxes

    def _traditional_nms(self, bounding_boxes, coefficients, scores,
                         iou_threshold=0.5, score_threshold=0.3, soft_nms_sigma=0.5):
        number_of_classes = tf.shape(scores)[1]

        number_of_coefficients = tf.shape(coefficients)[1]
        _bounding_boxes = tf.zeros(
            (self.per_class_max_output_size * number_of_classes, 4), tf.float32)
        _coefficients = tf.zeros(
            (self.per_class_max_output_size * number_of_classes), tf.float32)
        _classes = tf.zeros(
            (self.per_class_max_output_size * number_of_classes), tf.float32)
        _scores = tf.zeros(
            (self.per_class_max_output_size * number_of_classes), tf.float32)

        for _cls in range(number_of_classes):
            class_scores = scores[:, _cls]
            selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
                bounding_boxes,
                class_scores,
                max_output_size=self.per_class_max_output_size,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
                soft_nms_sigma=soft_nms_sigma)

            _updated_boxes = tf.gather(bounding_boxes, selected_indices)
            number_of_boxes = tf.shape(_updated_boxes)[0]
            box_indexes = tf.range(_cls * self.per_class_max_output_size,
                                   _cls * self.per_class_max_output_size + number_of_boxes)

            _bounding_boxes = tf.tensor_scatter_nd_update(
                _bounding_boxes, tf.expand_dims(box_indexes, axis=-1), _updated_boxes)
            _coefficients = tf.tensor_scatter_nd_update(_coefficients, tf.expand_dims(
                box_indexes, axis=-1), tf.gather(coefficients, selected_indices))
            _classes = tf.tensor_scatter_nd_update(_classes, tf.expand_dims(
                box_indexes, axis=-1), tf.gather(scores, selected_indices))
            _scores = tf.tensor_scatter_nd_update(_scores, tf.expand_dims(
                box_indexes, axis=-1), tf.gather(scores, selected_indices))

        _ids = tf.argsort(_scores, direction='DESCENDING')
        scores = tf.gather(_scores, _ids)[:self.max_output_size]
        bounding_boxes = tf.gather(_bounding_boxes, _ids)[
            :self.max_output_size]
        coefficients = tf.gather(_coefficients, _ids)[:self.max_output_size]
        classes = tf.gather(_classes, _ids)[:self.max_output_size]

        return (bounding_boxes, coefficients, classes, scores)
