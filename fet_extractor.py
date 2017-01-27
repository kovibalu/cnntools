import os

import numpy as np
from django.conf import settings

from cnntools.utils import add_caffe_to_path


def load_fet_extractor(
    deployfile_relpath,
    weights_relpath,
    do_preprocessing=True,
    image_dims=(256, 256),
    mean=(104, 117, 123),
    device_id=0,
    input_scale=1,
):
    # Silence Caffe
    from os import environ
    environ['GLOG_minloglevel'] = '2'

    add_caffe_to_path()
    import caffe

    FeatureExtractor = def_FeatureExtractor(caffe)

    if mean is not None:
        mean = np.array(mean)

    model_file = os.path.join(settings.CAFFE_ROOT, deployfile_relpath)
    pretrained_file = os.path.join(settings.CAFFE_ROOT, weights_relpath)

    if settings.CAFFE_GPU:
        print 'Using GPU'
        caffe.set_mode_gpu()
        print 'Using device #{}'.format(device_id)
        caffe.set_device(device_id)
    else:
        print 'Using CPU'
        caffe.set_mode_cpu()

    net = FeatureExtractor(
        model_file=model_file,
        pretrained_file=pretrained_file,
        do_preprocessing=do_preprocessing,
        image_dims=image_dims,
        mean=mean,
        input_scale=input_scale,
        raw_scale=255,
        channel_swap=(2, 1, 0),
    )

    return caffe, net


def def_FeatureExtractor(caffe):
    class FeatureExtractor(caffe.Net):
        """
        FeatureExtractor extends Net for to provide a simple interface for
        extracting features.

        Parameters
        ----------
        mean, input_scale, raw_scale, channel_swap: params for
            preprocessing options.
        """

        def __init__(self, model_file, pretrained_file, do_preprocessing, image_dims, mean=None,
                     input_scale=None, raw_scale=None, channel_swap=None):
            caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)
            self.do_preprocessing = do_preprocessing

            if not self.do_preprocessing:
                return

            # configure pre-processing
            in_ = self.inputs[0]
            self.transformer = caffe.io.Transformer(
                {in_: self.blobs[in_].data.shape})
            self.transformer.set_transpose(in_, (2, 0, 1))
            if mean is not None:
                self.transformer.set_mean(in_, mean)
            if input_scale is not None:
                self.transformer.set_input_scale(in_, input_scale)
            if raw_scale is not None:
                self.transformer.set_raw_scale(in_, raw_scale)
            if channel_swap is not None:
                self.transformer.set_channel_swap(in_, channel_swap)

            self.crop_dims = np.array(self.blobs[in_].data.shape[2:])
            self.image_dims = image_dims

        def preprocess_inputs(self, inputs, auto_reshape=True):
            """
            Preprocesses inputs.

            Parameters
            ----------
            inputs : iterable of (H x W x K) input ndarrays.

            Returns
            -------
            caffe_in: Preprocessed input which can be passed to forward.
            """
            # Only auto reshape if we don't set the image dimensions explicitly
            if auto_reshape and self.image_dims is None:
                # Keep original input dimensions and reshape the net
                # All inputs should have the same input dimensions!
                input_ = np.zeros(
                    (len(inputs), ) + inputs[0].shape,
                    dtype=np.float32
                )
                for ix, in_ in enumerate(inputs):
                    input_[ix] = in_
            else:
                # Scale to standardize input dimensions.
                input_ = np.zeros(
                    (
                        len(inputs), self.image_dims[0], self.image_dims[1],
                        inputs[0].shape[2]
                    ),
                    dtype=np.float32
                )
                for ix, in_ in enumerate(inputs):
                    input_[ix] = caffe.io.resize_image(in_, self.image_dims)

                # Take center crop.
                center = np.array(self.image_dims) / 2.0
                crop = np.tile(center, (1, 2))[0] + np.concatenate([
                    -self.crop_dims / 2.0,
                    self.crop_dims / 2.0
                ])
                input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]

            caffe_in = np.zeros(
                np.array(input_.shape)[[0, 3, 1, 2]],
                dtype=np.float32
            )
            if auto_reshape:
                self.reshape_by_input(caffe_in)

            for ix, in_ in enumerate(input_):
                caffe_in[ix] = self.transformer.preprocess(self.inputs[0], in_)

            return caffe_in

        def reshape_by_input(self, caffe_in):
            """
            Reshapes the whole net according to the input
            """
            print 'Reshaping net to {} input size...'.format(caffe_in.shape)
            in_ = self.inputs[0]
            self.blobs[in_].reshape(*caffe_in.shape)
            self.transformer.inputs = {in_: self.blobs[in_].data.shape}
            self.reshape()

        def predict(self, filename, auto_reshape=True):
            if isinstance(filename, np.ndarray):
                inputs = [filename]
            else:
                inputs = [caffe.io.load_image(filename)]

            if self.do_preprocessing:
                caffe_in = self.preprocess_inputs(inputs, auto_reshape=auto_reshape)
            else:
                # All inputs should have the same input dimensions!
                caffe_in = np.zeros(
                    (len(inputs), ) + inputs[0].shape,
                    dtype=np.float32
                )
                for ix, in_ in enumerate(inputs):
                    caffe_in[ix] = in_

            return self.forward_all(**{self.inputs[0]: caffe_in})

        def extract_features(self, filename, blob_names, auto_reshape=True):
            # sanity checking
            if len(set(blob_names)) != len(blob_names):
                raise ValueError("Duplicate name in blob_names: %s" % blob_names)

            self.predict(filename, auto_reshape=auto_reshape)
            ret = {}
            for blob_name in blob_names:
                blob_data = self.blobs[blob_name].data.copy()
                ret[blob_name] = blob_data

            return ret

    return FeatureExtractor
