import itertools
import json
import os
import random

from cnntools.common_utils import (ensuredir, iter_batch, make_prefix_dirs,
                                   progress_bar)
from cnntools.image_downloader import download_images
from django.conf import settings


def _gen_fgphoto_image_specs_func(data_path, items_split):
    '''Note that items_split is a QuerySet here'''
    items_data = items_split.values_list('id', 'photo_id')

    image_specs = []
    for item_id, photo_id in items_data:
        filename = '{}.jpg'.format(item_id)
        prefix_root_path = make_prefix_dirs(data_path, filename)
        filepath = os.path.join(prefix_root_path, filename)

        image_specs.append((item_id, [photo_id], [filepath]))

    return image_specs


def index_kwargs(kwargs, idx):
    if not kwargs:
        return {}

    return {k: v[idx] for k, v in kwargs.iteritems()}


def make_trainingfiles_simple_label(rel_root_path, filename_suffix, item_type,
                                    item_ids, skip_test, split_attr, y_true):
    """
    This function creates the training text files which is used for CNN
    training with Caffe. It also downloads all photos which are part of the
    dataset. This is a specialized function for generating simple one image /
    one label training files.

    :param rel_root_path: The root path of the photos and generated training
    files relative to the Caffe root path.

    :param filename_suffix: Added suffix to the generated training file names.

    :param item_type: The type of the model class for the item which are
    classified (e.g. FgPhoto). This class should have 'photo',
    'matclass_dataset_split' attributes/properties. The photo attribute should
    have most of the Photo model's fields. It is advised to use an actual Photo
    instance here. The matclass_dataset_split attribute should indicate
    which dataset split this item is in. The possible dataset splits are 'E'
    (test), 'V' (validation), 'R' (training).

    :param item_ids: List (or numpy array) of ids into the :ref:`item_type`
    table. The length of this list is the same as the length of :ref:`y_true`
    list. It should contain the training, validation and test set and have the
    same order as :ref:`y_true`.

    :param skip_test: If true, skip generating file and downloading images for
    the test split.

    :param split_attr: The attribute name which represents the dataset split in
    the database. It should be one character, 'E' meaning test, 'V' meaning
    validation, 'R' meaning training.

    :param y_true: List (or numpy array) of true labels for all items in the
    same order as the corresponding items in item_ids. Each element is an
    integer representing a label.
    """
    def gen_line_func(image_path, y_true):
        return ' '.join([image_path, str(y_true)])

    make_trainingfiles(
        rel_root_path, filename_suffix, item_type, item_ids, skip_test,
        split_attr,
        _gen_fgphoto_image_specs_func, gen_line_func, None,
        {'y_true': y_true}
    )


def make_trainingfiles_multi_tag(rel_root_path, filename_suffix, item_type,
                                 item_ids, skip_test, split_attr, tags_list):
    """
    This function creates the training text files which is used for CNN
    training with Caffe. It also downloads all photos which are part of the
    dataset. This is a specialized function for generating simple one image /
    one label training files.

    :param rel_root_path: The root path of the photos and generated training
    files relative to the Caffe root path.

    :param filename_suffix: Added suffix to the generated training file names.

    :param item_type: The type of the model class for the item which are
    classified (e.g. FgPhoto). This class should have 'photo',
    'matclass_dataset_split' attributes/properties. The photo attribute should
    have most of the Photo model's fields. It is advised to use an actual Photo
    instance here. The matclass_dataset_split attribute should indicate in
    which dataset split this item is in. The possible dataset splits are 'E'
    (test), 'V' (validation), 'R' (training).

    :param item_ids: List (or numpy array) of ids into the :ref:`item_type`
    table. The length of this list is the same as the length of :ref:`y_true`
    list. It should contain the training, validation and test set and have the
    same order as :ref:`y_true`.

    :param skip_test: If true, skip generating file and downloading images for
    the test split.

    :param split_attr: The attribute name which represents the dataset split in
    the database. It should be one character, 'E' meaning test, 'V' meaning
    validation, 'R' meaning training.

    :param tags_list: List of tag disctionaries for all items. Each element of
    tags_list is a dictionary of tag groups, in the same order as the
    corresponding items in item_ids. The dictionary is for different tag
    groups, the key is the tag group name and the value is a tag list. The tags
    are a list of integers each representing a tag. This is essentially a 3D
    array, but for each item and tag group, the number of tags can vary.
    """
    def gen_line_func(image_path, tags_list):
        line_dic = {
            'filepath': image_path,
            'tags_dic': tags_list,
        }
        # One line serialized to json
        return json.dumps(line_dic)

    make_trainingfiles(
        rel_root_path, filename_suffix, item_type, item_ids, skip_test,
        split_attr,
        _gen_fgphoto_image_specs_func, gen_line_func, None,
        {'tags_list': tags_list}
    )


def get_abbr_fname(skip_test):
    abbr = ['V', 'R']
    fnames = ['val', 'train']
    if not skip_test:
        abbr.insert(0, 'E')
        fnames.insert(0, 'test')

    return abbr, fnames


def process_images(rel_root_path, item_type, item_ids, skip_test, split_attr,
                   gen_image_specs_func, trafo_image_func,
                   trafo_image_extra_kwargs=None, img_obj_type=None,
                   img_attr=None, dimensions=(256, 256),
                   max_valset_size=10000):
    """
    This function downloads all photos which are part of the
    dataset. This is a general function which can be used for lots of different
    layers.

    It returns a dictionary which contains the downloaded image paths.
    Key: dataset split identifier, can be 'E', 'V', 'R'
    Value: tuple of (item indexes in the item_ids array, corresponding image paths)

    :param rel_root_path: The root path of the photos and generated training
    files relative to the Caffe root path.

    :param item_type: The type of the model class for the items which are
    classified (e.g. FgPhoto). This class should have 'photo',
    'matclass_dataset_split' attributes/properties. The photo attribute should
    have most of the Photo model's fields. It is advised to use an actual Photo
    instance here. The matclass_dataset_split attribute should indicate in
    which dataset split this item is in. The possible dataset splits are 'E'
    (test), 'V' (validation), 'R' (training).

    :param item_ids: List (or numpy array) of ids into the :ref:`item_type`
    table. It should contain the training, validation and test set.

    :param skip_test: If true, skip generating file and downloading images for
    the test split.

    :param split_attr: The attribute name which represents the dataset split in
    the database. It should be one character, 'E' meaning test, 'V' meaning
    validation, 'R' meaning training.

    :param gen_image_specs_func: Function which generates an id, photo id, image
    path triplet for each item which we later use to download the images.

    :param trafo_image_func: If None, we don't apply any transformation on the
    images. Function which transforms an image given the image path and the
    extra parameters, it should return the path of the transformed image, which
    can be the original image path or a new path.
    :ref:`trafo_image_extra_kwargs` will be passed as extra parameters to this function.

    :param trafo_image_extra_kwargs: Extra keyword arguments which will be passed to
    :ref:`trafo_image_func` function. All of them should be a list which has the
    same order as :ref:`item_ids`.

    :param img_obj_type: The type of the model class which holds an image.

    :param img_attr: The attribute of `img_obj_type` which holds the image.

    :param dimensions: The dimensions to resize the downloaded images to. If
    None, keep the image as original size.

    :param max_valset_size: The maximum size for the validation set.
    """
    item_id_to_idx = {id: idx for idx, id in enumerate(item_ids)}
    abbr, fnames = get_abbr_fname(skip_test)

    # The return value
    image_data = {}

    for mc_ds_s, fname in zip(abbr, fnames):
        data_path = os.path.join(rel_root_path, 'data')
        ensuredir(os.path.join(settings.CAFFE_ROOT, data_path))

        print 'Generating split file and downloading images for {} split...'.format(fname)
        print 'Generating a list of images to download...'
        image_specs = []
        for item_ids_batch in progress_bar(iter_batch(item_ids, 10000)):
            # Note that the order is not going to be the same as
            # item_ids_batch, so we expect the data layer to shuffle the data!
            items_split = (
                item_type.objects.
                filter(**{split_attr: mc_ds_s}).
                filter(id__in=item_ids_batch).
                order_by()
            )

            # A list of item_id, image_url, image_path tuples
            image_specs += gen_image_specs_func(data_path, items_split)

        if not image_specs:
            image_data[mc_ds_s] = ([], [])
            continue

        # We want the validation step to finish in tractable time, so we have a
        # maximum threshold on the validation set size
        if mc_ds_s == 'V' and len(image_specs) > max_valset_size:
            print 'Sampling {} images to reduce the size of the validation set...'.format(max_valset_size)
            # For reproducibility
            random.seed(125)
            image_specs = random.sample(image_specs, max_valset_size)

        item_ids_perm, img_obj_ids, image_paths_list = zip(*image_specs)

        # A corresponding list of indices into the item_ids array
        item_idxs = [item_id_to_idx[item_id] for item_id in item_ids_perm]

        # Add caffe root to all paths for downloading
        full_image_paths_list = [
            [
                os.path.join(settings.CAFFE_ROOT, ip)
                for ip in ipl
            ]
            for ipl in image_paths_list
        ]

        # Downloading images
        download_images(
            item_type=img_obj_type,
            item_ids=list(itertools.chain.from_iterable(img_obj_ids)),
            img_attr=img_attr,
            image_paths=list(itertools.chain.from_iterable(full_image_paths_list)),
            format='JPEG',
            dimensions=dimensions,
        )

        if trafo_image_func:
            print 'Transforming images...'
            new_image_paths_list = []
            new_item_idxs = []
            for item_idx, image_paths, full_image_paths in progress_bar(zip(item_idxs, image_paths_list, full_image_paths_list)):
                new_image_paths = trafo_image_func(
                    image_paths,
                    full_image_paths,
                    **index_kwargs(trafo_image_extra_kwargs, item_idx)
                )
                if not new_image_paths:
                    print ':( {}'.format(full_image_paths)
                    continue

                new_image_paths_list.append(new_image_paths)
                new_item_idxs.append(item_idx)

            image_paths_list = new_image_paths_list
            item_idxs = new_item_idxs

        image_data[mc_ds_s] = (item_idxs, image_paths_list)

    return image_data


def make_trainingfiles(rel_root_path, filename_suffix, item_type, item_ids,
                       skip_test, split_attr,
                       gen_image_specs_func, gen_line_func, trafo_image_func,
                       gen_line_extra_kwargs=None, trafo_image_extra_kwargs=None,
                       img_obj_type=None, img_attr=None, dimensions=(256, 256),
                       max_valset_size=10000):
    """
    This function creates the training text files which is used for CNN
    training with Caffe. It also downloads all photos which are part of the
    dataset. This is a general function which can be used for lots of different
    layers depending on the gen_line_func function.

    :param rel_root_path: The root path of the photos and generated training
    files relative to the Caffe root path.

    :param filename_suffix: Added suffix to the generated training file names.

    :param item_type: The type of the model class for the items which are
    classified (e.g. FgPhoto). This class should have 'photo',
    'matclass_dataset_split' attributes/properties. The photo attribute should
    have most of the Photo model's fields. It is advised to use an actual Photo
    instance here. The matclass_dataset_split attribute should indicate
    which dataset split this item is in. The possible dataset splits are 'E'
    (test), 'V' (validation), 'R' (training).

    :param item_ids: List (or numpy array) of ids into the :ref:`item_type`
    table. It should contain the training, validation and test set.

    :param skip_test: If true, skip generating file and downloading images for
    the test split.

    :param split_attr: The attribute name which represents the dataset split in
    the database. It should be one character, 'E' meaning test, 'V' meaning
    validation, 'R' meaning training.

    :param gen_image_specs_func: Function which generates an id, photo id, image
    path triplet for each item which we later use to download the images.

    :param gen_line_func: Function which generates a line into the training
    text file given the image path and the extra parameters.
    :ref:`gen_line_extra_kwargs` will be passed as extra parameters to this function.

    :param trafo_image_func: If None, we don't apply any transformation on the
    images. Function which transforms an image given the image path and the
    extra parameters, it should return the path of the transformed image, which
    can be the original image path or a new path.
    :ref:`trafo_image_extra_kwargs` will be passed as extra parameters to this function.

    :param gen_line_extra_kwargs: Extra keyword arguments which will be passed to
    :ref:`gen_line_func` function. All of them should be a list which has the
    same order as :ref:`item_ids`.

    :param trafo_image_extra_kwargs: Extra keyword arguments which will be passed to
    :ref:`trafo_image_func` function. All of them should be a list which has the
    same order as :ref:`item_ids`.

    :param img_obj_type: The type of the model class which holds an image.

    :param img_attr: The attribute of `img_obj_type` which holds the image.

    :param dimensions: The dimensions to resize the downloaded images to. If
    None, keep the image as original size.

    :param max_valset_size: The maximum size for the validation set.
    """
    image_data = process_images(
        rel_root_path=rel_root_path,
        item_type=item_type,
        item_ids=item_ids,
        skip_test=skip_test,
        split_attr=split_attr,
        gen_image_specs_func=gen_image_specs_func,
        trafo_image_func=trafo_image_func,
        trafo_image_extra_kwargs=trafo_image_extra_kwargs,
        img_obj_type=img_obj_type,
        img_attr=img_attr,
        dimensions=dimensions,
        max_valset_size=max_valset_size,
    )
    abbr, fnames = get_abbr_fname(skip_test)

    for mc_ds_s, fname in zip(abbr, fnames):
        splitfile_path = os.path.join(
            rel_root_path,
            '{}{}.txt'.format(fname, filename_suffix)
        )

        print 'Writing Caffe {} text file...'.format(fname)
        with open(os.path.join(settings.CAFFE_ROOT, splitfile_path), mode='w') as splitfile:
            item_idxs, image_paths_list = image_data[mc_ds_s]
            for item_idx, image_paths in progress_bar(zip(item_idxs, image_paths_list)):
                line = gen_line_func(
                    image_paths,
                    **index_kwargs(gen_line_extra_kwargs, item_idx)
                )
                splitfile.write('{}\n'.format(line))

