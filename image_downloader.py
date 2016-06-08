import time

from celery import current_app
from cnntools.common_utils import (iter_batch, progress_bar,
                                   progress_bar_widgets)
from photos.tasks import download_photo_batch_task
from progressbar import ProgressBar


def get_queue_info(queue_name):
    with current_app.broker_connection() as conn:
        with conn.channel() as channel:
            return channel.queue_declare(queue_name, passive=True)


def download_images(photo_ids, image_paths, format=None, dimensions=None):
    '''Downloads all images and saves them the the specified paths and returns
    the successes as a list. The image downloading will be dispatched on
    Celery and the results will be aggregated using a Queue.

    :param photo_ids: A list of Photo IDs to download.

    :param image_paths: A list of image paths where the images will be downloaded
    '''
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

