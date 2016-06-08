import json
import os

from django.conf import settings
from django.core.management.base import BaseCommand

from categories.models import NameCategory
from cnntools.cmd_utils import run_mincnet
from common.batch import progress_bar
from mattrans.dataset import NAMECAT_TO_DBNAME
from mattrans.models import MattransPhoto
from photos.add import add_photo_directory, deduplicate_photos
from photos.models import Photo


def save_photo_data(photos, photo_paths, photo_dirpath, desc_rootpath):
    print 'Saving photo id - path correspondences...'
    photo_data_dic = []
    for i, photo in enumerate(photos):
        if photo_dirpath:
            filename = os.path.basename(photo_paths[i])
            dirname = os.path.basename(os.path.dirname(photo_paths[i]))
        else:
            filename = None
            dirname = None

        photo_data_dic.append({
            'id': photo.id,
            'filename': filename,
            'dirname': dirname,
            'catname': photo.mattrans_photo.name_category.name,
            'textured': photo.mattrans_photo.textured,
            'shape_dataset': photo.mattrans_photo.shape.shape_dataset.name if photo.mattrans_photo.shape is not None else None,
            'unique_id': photo.mattrans_photo.shape.unique_id if photo.mattrans_photo.shape is not None else None,
            'camera_id': photo.mattrans_photo.camera_id,
        })
    json.dump(
        photo_data_dic,
        open(os.path.join(desc_rootpath, 'photo_data.json'), 'w')
    )


def comp_features(photo_dirpath, netid, snapshot_id, min_dim, fet_names,
                  rendered):
    for fet_name in fet_names:
        # Return the last one, because this should be the same...
        photos, photo_paths, desc_rootpath = comp_feature(
            photo_dirpath=photo_dirpath,
            netid=netid,
            snapshot_id=snapshot_id,
            min_dim=min_dim,
            fet_name=fet_name,
            rendered=rendered,
        )

    return photos, photo_paths, desc_rootpath


def comp_feature(photo_dirpath, netid, snapshot_id, min_dim, fet_name,
                 rendered):
    if rendered:
        slug_extra_add = '-rendered'
        desc_rootpath = os.path.join(settings.DATA_DIR, 'mattrans/results/descs/rendered')
    else:
        slug_extra_add = ''
        desc_rootpath = os.path.join(settings.DATA_DIR, 'mattrans/results/descs/real')

    no_transfer_netids = [
        'googlenet-minc'
    ]
    transfer_weights = netid not in no_transfer_netids

    grayscale = 'gray' in netid
    if grayscale:
        print 'The network will be evaluated on grayscale images!'

    #ids = [
        #9292651, 9405346, 9403845, 9295701, 9399785, 9400912, 9409393,
        #9404690, 9296788, 9403620, 9401360, 9402971, 9400426, 9298217,
        #9406908, 9292605, 9406643, 9296616, 9407740, 9399825, 9402814,
    #]
    name_cats = [
        'bed', 'chair', 'sofa', 'lamp', 'table',
    ]

    # If we specified a photo path then upload those images to the
    # database, if not, just get all items from the database
    if photo_dirpath:
        photos = []
        photo_paths = []
        for name_cat in name_cats:
            name_category = NameCategory.objects.get(
                name=NAMECAT_TO_DBNAME[name_cat]
            )
            cat_photo_dirpath = os.path.join(
                photo_dirpath,
                'preprocessed_%ss_images' % name_cat,
            )
            print 'Adding photos from directory: "%s"...' % cat_photo_dirpath
            cat_photos, cat_photo_paths = add_photo_directory(
                cat_photo_dirpath, {'in_mattrans_dataset': True}
            )

            item_ids, cat_photos, cat_photo_paths = deduplicate_photos(
                cat_photos, cat_photo_paths
            )
            print 'Adding MattransPhotos with the right category...'
            for p in progress_bar(cat_photos):
                MattransPhoto.objects.update_or_create(
                    defaults={'name_category': name_category},
                    photo=p,
                )

            photos += cat_photos
            photo_paths += cat_photo_paths
    else:
        print 'Getting photos from database...'
        photos = []
        photo_paths = None
        for name_cat in name_cats:
            name_category = NameCategory.objects.get(
                name=NAMECAT_TO_DBNAME[name_cat]
            )
            qset = (
                MattransPhoto.objects.
                filter(name_category=name_category).
                filter(photo__in_mattrans_dataset=True).
                filter(photo__aspect_ratio=1.0)
            )

            if rendered:
                qset = qset.filter(shape__shape_dataset__name='shapenet')
            else:
                qset = qset.filter(shape__isnull=True)

            photo_ids = qset.values_list('photo_id', flat=True)
            cat_photos = Photo.objects.in_bulk(photo_ids).values()
            photos += cat_photos

    image_trafo_kwargs, prob_desc_dir, probs = run_mincnet(
        photos=photos,
        netid=netid,
        snapshot_id=snapshot_id,
        min_dim=min_dim,
        grayscale=grayscale,
        transfer_weights=transfer_weights,
        fet_name=fet_name,
        slug_extra_add=slug_extra_add,
        desc_rootpath=desc_rootpath,
    )

    return photos, photo_paths, desc_rootpath


class Command(BaseCommand):
    args = ''
    help = (
        'Runs the MINC Net on a directory of images and generates blobby '
        'predictions. This method supports only the ".jpg" extension and all '
        'images should have the same resolution.'
    )

    option_list = BaseCommand.option_list + (
    )

    def handle(self, *args, **option):
        # Rootpath specifies the directory where we will save the features
        #netid = args[0]
        #min_dim = int(args[1])
        #photo_dirpath = args[2]
        #result_dirpath = args[3]
        #if len(args) > 4:
            #snapshot_id = int(args[4])
        #else:
            #snapshot_id = None
        #if len(args) > 5:
            #fet_name = args[5]
        #else:
            #fet_name = 'prob'

        configs = [
            {
                'netid': 'googlenet-minc',
                'snapshot_id': 64,
                'min_dim': 900,
                'rendered': False,
                'fet_names': ['prob'],
            },

            # Old
            #{
                #'netid': 'googlenet-minc-mindim600-patch256-gray',
                #'snapshot_id': 286,
                #'min_dim': 400,
            #},
            #{
                #'netid': 'googlenet-minc-mindim600-patch256-gray',
                #'snapshot_id': 293,
                #'min_dim': 400,
            #},
            #{
                #'netid': 'googlenet-minc-mindim600-patch256',
                #'snapshot_id': 196,
                #'min_dim': 900,
            #},

            # New, balanced
            #{
                #'netid': 'googlenet-minc-mindim600-patch256-gray',
                #'snapshot_id': 406,
                #'min_dim': 900,
                #'rendered': True,
                #'fet_names': ['prob'],
            #},
            #{
                #'netid': 'googlenet-minc-mindim600-patch256-gray',
                #'snapshot_id': 406,
                #'min_dim': 900,
                #'rendered': False,
                #'fet_names': ['inception_5b/output'],
            #},
            #{
                #'netid': 'googlenet-minc-mindim900-patch256',
                #'snapshot_id': 410,
                #'min_dim': 900,
                #'rendered': False,
                #'fet_names': ['prob', 'inception_5b/output'],
            #},
        ]
        for kwargs in configs:
            photos, photo_paths, desc_rootpath = comp_features(
                photo_dirpath=None,
                **kwargs
            )

        save_photo_data(
            photos=photos,
            photo_paths=photo_paths,
            photo_dirpath=None,
            desc_rootpath=desc_rootpath,
        )

