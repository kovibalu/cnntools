import os

import numpy as np

from celery import shared_task
from cnntools import packer
from cnntools.common_utils import progress_bar, import_function
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


@shared_task(queue='gpu-train')
def start_training_task(netid, options):
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

    if options['local']:
        device_id = 0
    else:
        device_id = get_worker_gpu_device_id()

    start_training(
        model_name=netid,
        model_file_content=model_file_content,
        solver_file_content=solver_file_content,
        options=options,
        caffe_cnn_trrun_id=caffe_cnn_trrun_id,
        device_id=device_id,
    )


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
    which we use the retrieve the images the feature will be computed on

    :param task_id: ID of the task which will be used as a key to put the
    batch_id as a completed ID in redis

    :param batch_id: ID of the batch which will be used as a key to put the
    results in redis

    :param id_list: List of item_type IDs

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
    items = item_type.objects.in_bulk(id_list).values()

    fet_trafo_types = get_fet_trafo_types()

    fets = []
    print 'Computing features for {} items...'.format(len(items))
    for item in progress_bar(items):
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

