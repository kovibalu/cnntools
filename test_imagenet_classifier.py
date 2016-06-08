import os

import numpy as np
from django.conf import settings


def get_imagenet_labels():
    imagenet_labels_filename = os.path.join(settings.CAFFE_ROOT, 'data/ilsvrc12/synset_words.txt')
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

    return labels


def test_imagenet_classifier(classifier):
    print 'Testing classifier...'
    blob_names = [
        'prob',
    ]
    img = os.path.join(settings.CAFFE_ROOT, 'examples/images/cat.jpg')
    fetdic = classifier.extract_features(img, blob_names=blob_names)

    labels = get_imagenet_labels()
    inds = np.argsort(np.squeeze(fetdic['prob']))[-1:-6:-1]
    print inds
    print labels[inds]
