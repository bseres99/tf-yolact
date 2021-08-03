import os

import tensorflow as tf

from anchor import Anchor
from data.yolact_parser import Parser


def prepare_dataset(
        image_height,
        image_width,
        feature_map_sizes,
        protonet_out_sizes,
        aspect_ratios,
        scales,
        tfrecord_directory,
        batch_size,
        subset="train"):

    anchorobject = Anchor(image_width=image_width, image_height=image_height,
                          feature_map_sizes=feature_map_sizes,
                          aspect_ratios=aspect_ratios, scales=scales)

    parser = Parser(output_size=[image_height, image_width],
                    anchor_instance=anchorobject,
                    match_threshold=0.5,
                    unmatched_threshold=0.5,
                    mode=subset,
                    prototype_output_size=[int(protonet_out_sizes[0]), int(protonet_out_sizes[1])])
    
    files = tf.io.matching_files(os.path.join(tfrecord_directory, "*.*"))
    number_of_shards = tf.cast(tf.shape(files)[0], tf.int64)
    
    shards = tf.data.Dataset.from_tensor_slices(files)
    shards = shards.shuffle(number_of_shards)
    shards = shards.repeat()

    dataset = shards.interleave(tf.data.TFRecordDataset,
                                cycle_length=number_of_shards,
                                num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=2048)
    dataset = dataset.map(map_func=parser, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset
