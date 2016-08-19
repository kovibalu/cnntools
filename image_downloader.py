import time

from celery import current_app
from cnntools.common_utils import (iter_batch, progress_bar,
                                   progress_bar_widgets)
from progressbar import ProgressBar


def get_queue_info(queue_name):
    with current_app.broker_connection() as conn:
        with conn.channel() as channel:
            return channel.queue_declare(queue_name, passive=True)


def download_photos(photo_ids, image_paths, format=None, dimensions=None):
    '''This is a specialized function which handles only Photo objects.
    Downloads all images and saves them the the specified paths and returns the
    successes as a list. The image downloading will be dispatched on Celery and
    the results will be aggregated using a Queue.

    :param photo_ids: A list of Photo IDs to download.

    :param image_paths: A list of image paths where the images will be downloaded
    '''
    from photos.tasks import download_photo_batch_task
    print 'Dispatching tasks...'
    batch_num = 0
    for batch in iter_batch(progress_bar(zip(photo_ids, image_paths)), n=1024):
        photo_ids_batch, image_paths_batch = zip(*batch)
        download_photo_batch_task.delay(
            photo_ids_batch, image_paths_batch, format, dimensions
        )
        batch_num += 1

    # TODO: Maybe put a wait here to let all the tasks be dispatched, because
    # we may quit before they are all finished...

    print 'Waiting for tasks to finish...'
    pbar = ProgressBar(
        widgets=progress_bar_widgets(),
        maxval=batch_num,
    )
    pbar.start()

    while True:
        time.sleep(5)
        queue_info = get_queue_info('artifact')
        pbar.update(batch_num - queue_info.message_count)

        if queue_info.message_count == 0:
            break

    pbar.finish()


def download_images(item_type, item_ids, img_attr, image_paths, format=None, dimensions=None):
    '''Downloads all images and saves them the the specified paths and returns
    the successes as a list. The image downloading will be dispatched on
    Celery and the results will be aggregated using a Queue.

    :param item_type: The type of the model class for the items which contain
    the image data we want to download. This class should have `img_attr`
    attribute/property.

    :param item_ids: A list of IDs of database objects which have the location of the images to download.

    :param img_attr: The name of the attribute of the database object which
    stores the image to be downloaded. Note: This field has to be a django
    ImageField (or behave like one).

    :param image_paths: A list of image paths where the images will be downloaded
    '''
    from cnntools.tasks import download_image_batch_task
    print 'Dispatching tasks...'
    batch_num = 0
    for batch in iter_batch(progress_bar(zip(item_ids, image_paths)), n=1024):
        item_ids_batch, image_paths_batch = zip(*batch)
        download_image_batch_task.delay(
            item_type, item_ids_batch, image_paths_batch, img_attr, format,
            dimensions
        )
        batch_num += 1

    # TODO: Maybe put a wait here to let all the tasks be dispatched, because
    # we may quit before they are all finished...

    print 'Waiting for tasks to finish...'
    pbar = ProgressBar(
        widgets=progress_bar_widgets(),
        maxval=batch_num,
    )
    pbar.start()

    while True:
        time.sleep(5)
        queue_info = get_queue_info('artifact')
        pbar.update(batch_num - queue_info.message_count)

        if queue_info.message_count == 0:
            break

    pbar.finish()

