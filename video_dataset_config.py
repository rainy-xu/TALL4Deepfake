

DATASET_CONFIG = {
    'ffpp': {
        'num_classes': 2,
        'train_list_name': 'cdf_test_fold.txt',
        'val_list_name': 'cdf_test_fold.txt',
        'test_list_name': 'cdf_test_fold.txt',
        'filename_seperator': " ",
        'image_tmpl': '{:04d}.jpg',
        'filter_video': 3,
    }
}


def get_dataset_config(dataset, use_lmdb=False):
    ret = DATASET_CONFIG[dataset]
    num_classes = ret['num_classes']
    train_list_name = ret['train_list_name'].replace("txt", "lmdb") if use_lmdb \
        else ret['train_list_name']
    val_list_name = ret['val_list_name'].replace("txt", "lmdb") if use_lmdb \
        else ret['val_list_name']
    test_list_name = ret['test_list_name'].replace("txt", "lmdb") if use_lmdb \
        else ret['test_list_name']
    filename_seperator = ret['filename_seperator']
    image_tmpl = ret['image_tmpl']
    filter_video = ret.get('filter_video', 0)
    label_file = ret.get('label_file', None)

    return num_classes, train_list_name, val_list_name, test_list_name, filename_seperator, \
           image_tmpl, filter_video, label_file
