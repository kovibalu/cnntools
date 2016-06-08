import os

import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand
from scipy.misc import imsave

from cnntools.fetcomp import compute_crf_features
from cnntools.trafos import minc_padding
from common.batch import progress_bar
from common.utils import ensuredir
from matclass import accuracy
from matclass.dataset import NAME_TO_NETCAT, NLABELS
from photos.add import add_photo_directory, deduplicate_photos
from photos.models import Photo


class Command(BaseCommand):
    args = ''
    help = (
        'Runs the MINC Net on a directory of images and generates CRF corrected predictions from the give blobby predictions. '
        'predictions. This method supports only the ".jpg" extension and all '
        'images should have the same resolution.'
    )

    option_list = BaseCommand.option_list + (
    )

    def handle(self, *args, **option):
        desc_rootpath = settings.DESC_ROOTPATH
        # The directory which contains the computed probabilities
        prob_desc_dir = args[0]
        photo_dirpath = args[1]
        result_dirpath = args[2]
        slug = args[3]

        save_images = False
        photos, photo_paths = add_photo_directory(photo_dirpath)
        item_ids, photos, photo_paths = deduplicate_photos(photos, photo_paths)

        params = {
            "caffemodel": "230-iter230000.caffemodel",
            "deploy_source": "230-deploy.prototxt",
            "deploy_target": "230-deploy-conv.prototxt",
            "effective_stride": 32,
            "input_pad": 96,
            "max_dim": None,
            "min_dim": 600,
            "mirror": False,
            "mode": "sliding",
            #"oversample_pad": 2,
            #"oversample_scale": 3,
            "oversample_pad": 1,
            "oversample_scale": 1,
            "param_mapping": {"fc8-conv": "fc8-20"},
            "receptive_field": 224,
        }
        image_trafo_kwargs = {
            'params': params,
        }
        crf_params = {
            "bilateral_pairwise_weight": 8,
            "bilateral_theta_lab_ab": 3.0,
            "bilateral_theta_lab_l": 0.5,
            "bilateral_theta_xy": 0.5,
            "min_dim": 550,
            "n_crf_iters": 10,
            "splat_triangle_weight": 1,
            "unary_prob_padding": 1e-05,
            "ignore_labels": [NAME_TO_NETCAT['other']],
            "stride": params['effective_stride'],
        }

        # Compute the expected output size
        padded_shape = minc_padding(photos[0], **image_trafo_kwargs).shape
        pixel_count = np.prod(padded_shape[:2])

        layer_params = [
            ('crf', pixel_count),
        ]
        feature_name_list, num_dims_list = zip(*layer_params)

        tmp_desc_rootpath = os.path.join(desc_rootpath, 'temp')
        ensuredir(tmp_desc_rootpath)
        ensuredir(result_dirpath)

        fets = compute_crf_features(
            desc_rootpath=tmp_desc_rootpath,
            item_type=Photo,
            item_ids=item_ids,
            node_batchsize=1,
            aggr_batchsize=8,
            feature_name_list=feature_name_list,
            num_dims_list=num_dims_list,
            crf_params=crf_params,
            label_count=NLABELS,
            desc_dir=prob_desc_dir,
            slug=slug,
            image_trafo_type_id='MINC-padding',
            image_trafo_kwargs=image_trafo_kwargs,
        )

        labels_crf_list = fets['crf']

        if save_images:
            print 'Saving photos with predictions...'
            for photo, photo_path, labels_crf in progress_bar(zip(photos, photo_paths, labels_crf_list)):
                fname = os.path.basename(photo_path)
                res_photo_path = os.path.join(result_dirpath, 'res-crf-' + fname)

                labels_crf = np.reshape(labels_crf, padded_shape[:2]).astype(int)
                orig_img = minc_padding(photo, **image_trafo_kwargs)
                # Grayscale
                orig_img = np.mean(orig_img, axis=2)[:, :, np.newaxis]
                label_img = accuracy.labels_to_color(labels_crf)
                final_img = orig_img * 255 * 0.5 + label_img * 0.5
                imsave(res_photo_path, final_img)

