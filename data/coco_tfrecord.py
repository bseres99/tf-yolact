import tensorflow as tf
import io
import os
import json
import PIL.Image
import hashlib
from pycocotools import mask
import dataset_utils
import numpy as np
import contextlib
import logging


def create_tf_example(image,
                      annotations_list,
                      image_directory,
                      category_index,
                      include_masks=True):
    image_height = image['height']
    image_width = image['width']
    filename = image['file_name']
    image_id = image['id']

    full_path = os.path.join(image_directory, filename)
    with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    r = 550
    image = image.resize((r, r), PIL.Image.ANTIALIAS)
    bytes_io = io.BytesIO()
    image.save(bytes_io, format='JPEG')
    encoded_jpg = bytes_io.getvalue()
    key = hashlib.sha256(encoded_jpg).hexdigest()

    xmin, \
        xmax, \
        ymin, \
        ymax, \
        is_crowd, \
        category_names, \
        category_ids, \
        areas, \
        encoded_mask_pngs = []
    number_of_annotations_skipped = 0

    for object_annotations in annotations_list:
        (x, y, width, height) = tuple(object_annotations['bbox'])
        if width <= 0 or height <= 0:
            number_of_annotations_skipped += 1
            continue
        if x + width > image_width or y + height > image_height:
            number_of_annotations_skipped += 1
            continue

        xmin.append(float(x) / image_width)
        xmax.append(float(x + width) / image_width)
        ymin.append(float(y) / image_height)
        ymax.append(float(y + height) / image_height)

        is_crowd.append(object_annotations['iscrowd'])
        category_id = int(object_annotations['category_id'])
        category_ids.append(category_id)
        category_names.append(
            category_index[category_id]['name'].encode('utf8'))
        areas.append(object_annotations['area'])

        if include_masks:
            run_len_encoding = mask.frPyObjects(object_annotations['segmentation'],
                                                image_height, image_width)
            binary_mask = mask.decode(run_len_encoding)
            if not object_annotations['iscrowd']:
                binary_mask = np.amax(binary_mask, axis=2)

            pil_image = PIL.Image.fromarray(binary_mask)
            output_io = io.BytesIO()
            pil_image.save(output_io, format='PNG')
            encoded_mask_pngs.append(output_io.getvalue())

        feature_dictionary = {
            'image/height':
                dataset_utils.int64_feature(r),
            'image/width':
                dataset_utils.int64_feature(r),
            'image/filename':
                dataset_utils.bytes_feature(filename.encode('utf8')),
            'image/source_id':
                dataset_utils.bytes_feature(str(image_id).encode('utf8')),
            'image/key/sha256':
                dataset_utils.bytes_feature(key.encode('utf8')),
            'image/encoded':
                dataset_utils.bytes_feature(encoded_jpg),
            'image/format':
                dataset_utils.bytes_feature('jpeg'.encode('utf8')),
            'image/object/bbox/xmin':
                dataset_utils.float_list_feature(xmin),
            'image/object/bbox/xmax':
                dataset_utils.float_list_feature(xmax),
            'image/object/bbox/ymin':
                dataset_utils.float_list_feature(ymin),
            'image/object/bbox/ymax':
                dataset_utils.float_list_feature(ymax),
            'image/object/class/text':
                dataset_utils.bytes_list_feature(category_names),
            'image/object/class/label':
                dataset_utils.int64_list_feature(category_ids),
            'image/object/is_crowd':
                dataset_utils.int64_list_feature(is_crowd),
            'image/object/area':
                dataset_utils.float_list_feature(areas),
        }

        if include_masks:
            feature_dictionary['image/object/mask'] = (
                dataset_utils.bytes_list_feature(encoded_mask_pngs))

        example = tf.train.Example(
            features=tf.train.Feature(feature=feature_dictionary))

        return key, example, number_of_annotations_skipped


def create_tf_record_from_coco_annotations(annotations_file,
                                           image_directory,
                                           output_path,
                                           include_masks,
                                           number_of_shards):
    with contextlib.ExitStack() as tf_record_close_stack, \
            tf.io.gfile.GFile(annotations_file, 'r') as fid:

        output_tfrecords = dataset_utils.open_sharded_output_tfrecords(
            tf_record_close_stack, output_path, number_of_shards)
        groundtruth_data = json.load(fid)
        images = groundtruth_data['images']
        category_index = dataset_utils.create_category_index(
            groundtruth_data['categories'])

        annotations_index = {}
        if 'annotations' in groundtruth_data:
            logging.info(
                'Found groundtruth annotations. Building annotation indices.')
            for annotation in groundtruth_data['annotations']:
                image_id = annotation['image_id']
                if image_id not in annotations_index:
                    annotations_index[image_id] = []
                annotations_index[image_id].append(annotation)

        missing_annotation_count = 0
        for image in images:
            image_id = image['id']
            if image_id not in annotations_index:
                missing_annotation_count += 1
                annotations_index[image_id] = []
        logging.info('%d images are missing annotations.',
                     missing_annotation_count)

        total_number_of_annotations_skipped = 0
        for idx, image in enumerate(images):
            if idx % 100 == 0:
                logging.info('On image %d of %d', idx, len(images))
            annotations_list = annotations_index[image['id']]

            number_of_crowds = 0
            for object_annotations in annotations_list:
                if object_annotations['iscrowd']:
                    number_of_crowds += 1

            if number_of_crowds != len(annotations_list):
                _, tf_example, number_of_annotations_skipped = create_tf_example(
                    image, annotations_list,
                    image_directory, category_index, include_masks)

                total_number_of_annotations_skipped += number_of_annotations_skipped
                shard_idx = idx % number_of_shards
                output_tfrecords[shard_idx].write(
                    tf_example.SerializeToString())
            else:
                logging.info('Image only have crowd annotation ignored')
                total_number_of_annotations_skipped += len(annotations_list)

        logging.info('Finished writing, skipped %d annotations.',
                     total_number_of_annotations_skipped)
