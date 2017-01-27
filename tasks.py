import os

import numpy as np

import redis
from celery import shared_task
from cnntools import packer
from cnntools.common_utils import ensuredir, import_function, progress_bar
from cnntools.fet_extractor import load_fet_extractor
from cnntools.models import CaffeCNN
from cnntools.redis_aggregator import batch_ready
from cnntools.snapshot_utils import download_snapshot
from cnntools.timer import Timer
from cnntools.trafos import gram_fets, mean_std_fets, spatial_avg_fets
from cnntools.trainer import start_training
from cnntools.utils import create_default_trrun, get_worker_gpu_device_id
from django.conf import settings


def get_fet_trafo_types():
    return {
        'MINC-spatial-avg': spatial_avg_fets,
        'MINC-gram': gram_fets,
        'MINC-mean-std': mean_std_fets,
    }


def schedule_training(netid, options):
    caffe_cnn = CaffeCNN.objects.get(netid=netid)
    model_file_content = caffe_cnn.get_model_file_content()
    solver_file_content = caffe_cnn.get_solver_file_content()
    deploy_file_content = caffe_cnn.get_deploy_file_content()

    if 'caffe_cnn_trrun_id' in options and options['caffe_cnn_trrun_id'] is not None:
        # This should be the id of an existing training run!
        caffe_cnn_trrun_id = options['caffe_cnn_trrun_id']
    else:
        # Snapshot model and solver files
        caffe_cnn_trrun = create_default_trrun(
            caffe_cnn,
            model_file_content,
            solver_file_content,
            deploy_file_content,
            description=options['description'],
        )
        caffe_cnn_trrun_id = caffe_cnn_trrun.id

    if 'verbose' not in options:
        options['verbose'] = False

    if options['local']:
        task_func = start_training_task
    else:
        task_func = start_training_task.delay

    task_func(
        model_name=netid,
        model_file_content=model_file_content,
        solver_file_content=solver_file_content,
        options=options,
        caffe_cnn_trrun_id=caffe_cnn_trrun_id,
    )

    return caffe_cnn_trrun_id


@shared_task(queue='gpu-train')
def start_training_task(model_name, model_file_content, solver_file_content,
                        options, caffe_cnn_trrun_id):
    if options['local']:
        device_id = 0
    else:
        device_id = get_worker_gpu_device_id()

    start_training(
        model_name=model_name,
        model_file_content=model_file_content,
        solver_file_content=solver_file_content,
        options=options,
        caffe_cnn_trrun_id=caffe_cnn_trrun_id,
        device_id=device_id,
    )


class RedisItem():
    def __init__(self, key, value):
        from cnntools.fetcomp import RedisItemKey
        self.item_key = RedisItemKey.create_from_key(key)
        self.value = value

    @property
    def id(self):
        return self.item_key.item_id


@shared_task(queue='gpu')
def compute_cnn_features_gpu_task(
    item_type,
    task_id,
    batch_id,
    id_list,
    feature_name_list,
    kwa,
):
    '''
    Computes the features for a list of model_class type objects (for the
    associated images), then sends the computed features to the redis server.
    The accumulator script will collect these and save them as a numpy array on
    disk.

    :param item_type: The class for the model which holds the information
    which we use the retrieve the images the feature will be computed on. This
    can also be 'redis', which means that we will fetch the item information
    from redis.

    :param task_id: ID of the task which will be used as a key to put the
    batch_id as a completed ID in redis

    :param batch_id: ID of the batch which will be used as a key to put the
    results in redis

    :param id_list: List of item_type IDs. If ``item_type`` is 'redis', these
    should be the redis keys corresponding to the items.

    :param feature_name_list: The features' name in the network which will be
    extracted

    :param kwa: The parameters to pass to the feature computer
    function.
    '''
    # Change working directory to Caffe
    os.chdir(settings.CAFFE_ROOT)

    device_id = get_worker_gpu_device_id()
    deployfile_relpath, weights_relpath = download_snapshot(
        kwa['snapshot_id'], kwa['transfer_weights']
    )
    caffe, fet_extractor = load_fet_extractor(
        deployfile_relpath, weights_relpath, kwa['do_preprocessing'],
        kwa['image_dims'], kwa['mean'], device_id
    )
    # This doesn't preserve order!
    if item_type == 'redis':
        client = redis.StrictRedis(**settings.REDIS_AGGRO_LOCAL_CONFIG)
        redis_vals = client.mget(*id_list)
        client.delete(*id_list)
        items = [RedisItem(key, value) for key, value in zip(id_list, redis_vals)]
    else:
        items = item_type.objects.in_bulk(id_list).values()

    fet_trafo_types = get_fet_trafo_types()

    fets = []
    print 'Computing features for {} items...'.format(len(items))
    show_progress = False
    if show_progress:
        items = progress_bar(items)

    for item in items:
        if kwa['input_trafo_func_name']:
            with Timer('Input transformation'):
                input_trafo = import_function(kwa['input_trafo_func_name'])
                inp = input_trafo(item, **kwa['input_trafo_kwargs'])
        else:
            inp = caffe.io.load_image(item.photo.image_300)

        if 'grayscale' in kwa and kwa['grayscale']:
            inp_gray = np.mean(inp, axis=2)
            inp = np.zeros_like(inp)
            inp[:, :, :] = inp_gray[:, :, np.newaxis]

        with Timer('Feature extraction'):
            # feature_name_list can contain 'img', which means that we want to
            # save the img or some transformation of the image as a final
            # feature result. Of course the CNN doesn't need to compute this
            # feature.
            fnl = list(feature_name_list)
            if 'img' in feature_name_list:
                fnl.remove('img')

            fetdic = fet_extractor.extract_features(
                inp, blob_names=fnl, auto_reshape=kwa['auto_reshape']
            )

        if kwa['fet_trafo_type_id']:
            with Timer('Feature transformation'):
                # This might add 'img' the the feature list
                fetdic = fet_trafo_types[kwa['fet_trafo_type_id']](
                    item, inp, fetdic, feature_name_list,
                    **kwa['fet_trafo_kwargs']
                )

        for feature_name in feature_name_list:
            fetdic[feature_name] = np.ravel(np.squeeze(fetdic[feature_name]))
        fets.append((item.id, fetdic))

    # Save results in redis
    batch_ready(task_id, batch_id, packer.packb(fets, settings.API_VERSION))


def _save_image(img_file, filename, format, dimensions):
    import shutil
    from PIL import Image
    from imagekit.utils import open_image
    from pilkit.utils import extension_to_format
    from cnntools.common_utils import resize_mindim

    # Skip if we already downloaded the image
    if os.path.exists(filename):
        return

    parent_dir = os.path.dirname(filename)
    ensuredir(parent_dir)

    if not dimensions and not format:
        img_file.seek(0)
        with open(filename, 'wb') as f:
            shutil.copyfileobj(img_file, f)
    else:
        if dimensions and len(dimensions) == 2:
            image = open_image(img_file)

            if image.size != tuple(dimensions):
                image = image.resize(dimensions, Image.ANTIALIAS)
        elif dimensions and len(dimensions) == 1:
            # Here we specified the minimum dimension
            image = open_image(img_file)
            image = resize_mindim(image, dimensions[0])
        else:
            image = open_image(img_file)

        if not format:
            format = extension_to_format(os.path.splitext(filename)[1].lower())

        image.save(filename, format)


@shared_task(queue='artifact')
def download_image_batch_task(item_type, item_ids, filenames, img_attr, format=None, dimensions=None):
    """ Downloads an image and stores it, potentially downsampling it and
    potentially converting formats """
    item_dic = item_type.objects.in_bulk(item_ids)
    for item_id, filename in zip(item_ids, filenames):
        _save_image(
            img_file=getattr(item_dic[item_id], img_attr),
            filename=filename,
            format=format,
            dimensions=dimensions,
        )
