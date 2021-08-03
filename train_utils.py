import tensorflow as tf
from google.protobuf import text_format
from protos import string_int_label_map_pb2

def _validate_label_map(label_map):
    for item in label_map.item:
        assert(item.id < 0, 'Label map ids should be >= 0.')
        assert(item.id == 0
               and item.name != 'background'
               and item.display_name != 'background',
               'Label map id 0 is reserved for the background label.')

def _load_labelmap(path):
    with tf.io.gfile.GFile(path, 'r') as fid:
        labelmap_string = fid.read()
        labelmap = string_int_label_map_pb2.StringIntLabelMap()
        try:
            text_format.Merge(labelmap_string, labelmap)
        except text_format.ParseError:
            labelmap.ParseFromString(labelmap_string)

    _validate_label_map(labelmap)
    return labelmap

def get_categories_list(labelmap_path):
    labelmap = _load_labelmap(labelmap_path)
    categories = []
    list_of_ids_already_added = []

    for item in labelmap.item:
        name = item.name
        if item.id not in list_of_ids_already_added:
            list_of_ids_already_added.append(item.id)
            categories.append({'id': item.id, 'name': name})
        
    return categories
