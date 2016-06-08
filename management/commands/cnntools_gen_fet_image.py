import os

import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand

from cnntools.deepart.deepart import gen_target_data, optimize_img
from cnntools.deepart.test_deepart import test_all_gradients
from cnntools.fet_extractor import load_fet_extractor
from common.utils import ensuredir


def setup_classifier_default():
    deployfile_relpath = os.path.join(
        settings.CAFFE_ROOT,
        'models/VGG_CNN_19/VGG_ILSVRC_19_layers_deploy_fullconv.prototxt'
    )
    weights_relpath = os.path.join(
        settings.CAFFE_ROOT,
        #'models/VGG_CNN_19/VGG_ILSVRC_19_layers.caffemodel'
        'models/VGG_CNN_19/vgg_normalised.caffemodel'
    )
    image_dims = (333, 500)
    #image_dims = (256, 256)
    #image_dims = (2, 2)
    #mean = (104, 117, 123)
    mean = (103.939, 116.779, 123.68)
    device_id = 0

    caffe, net = load_fet_extractor(
        deployfile_relpath, weights_relpath, image_dims, mean, device_id
    )
    #test_imagenet_classifier(classifier)

    return caffe, net, image_dims


class Command(BaseCommand):
    args = ''
    help = ''

    def handle(self, *args, **option):
        np.random.seed(123)

        root_dir = 'gen_fet_image_debug'
        ensuredir(root_dir)
        display = 100
        max_iter = 100000
        targets = [
            (os.path.join(settings.CAFFE_ROOT, 'examples/images/starry_night.jpg'), ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'], True, 100),
            (os.path.join(settings.CAFFE_ROOT, 'examples/images/tuebingen.jpg'), ['conv4_2'], False, 1),
            #os.path.join(settings.CAFFE_ROOT, 'examples/images/cat.jpg'),
            #os.path.join(settings.CAFFE_ROOT, 'examples/images/horse-head.jpg'),
        ]
        # These have to be in the same order as in the network!
        all_target_blob_names = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2', 'conv5_1']
        #all_target_blob_names = ['conv1_1', 'conv2_1']
        #all_target_blob_names = ['conv4_2']

        caffe, net, image_dims = setup_classifier_default()

        # Generate activations for input images
        target_data_list = gen_target_data(root_dir, caffe, net, targets)

        # Generate white noise image
        init_img = np.random.normal(loc=0.5, scale=0.1, size=image_dims + (3,))

        solver_type = 'L-BFGS-B'
        solver_param = {
            'maxcor': 20,
            'maxfun': 15000,
        }
        #test_all_gradients(init_img, net, all_target_blob_names, targets, target_data_list)

        optimize_img(
            init_img, solver_type, solver_param, max_iter, display, root_dir, net,
            all_target_blob_names, targets, target_data_list
        )

