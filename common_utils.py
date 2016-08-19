# Progressbar tools by Sean Bell
import hashlib
import os
import uuid
from tempfile import NamedTemporaryFile

from django.conf import settings
from django.contrib.contenttypes.models import ContentType
from django.db import models
from django.db.models.query import QuerySet
from django.utils.deconstruct import deconstructible
from django.utils.timezone import now
from progressbar import (AdaptiveETA, Bar, FileTransferSpeed, ProgressBar,
                         SimpleProgress)
from queued_storage.utils import import_attribute


def progress_bar(l, show_progress=True):
    """ Returns an iterator for a list or queryset that renders a progress bar
    with a countdown timer """
    if show_progress:
        if isinstance(l, QuerySet):
            return queryset_progress_bar(l)
        else:
            return iterator_progress_bar(l)
    else:
        return l


def progress_bar_widgets(unit='items'):
    return [
        SimpleProgress(sep='/'), ' ', Bar(), ' ',
        FileTransferSpeed(unit=unit), ', ', AdaptiveETA()
    ]


def iterator_progress_bar(iterator, maxval=None):
    """ Returns an iterator for an iterator that renders a progress bar with a
    countdown timer """

    if maxval is None:
        try:
            maxval = len(iterator)
        except:
            return iterator

    if maxval > 0:
        pbar = ProgressBar(maxval=maxval, widgets=progress_bar_widgets())
        return pbar(iterator)
    else:
        return iterator


def queryset_progress_bar(queryset):
    """ Returns an iterator for a queryset that renders a progress bar with a
    countdown timer """
    return iterator_progress_bar(queryset.iterator(), maxval=queryset.count())


def iter_batch(iterable, n=1024):
    """ Group an iterable into batches of size n """
    if hasattr(iterable, '__getslice__'):
        for i in xrange(0, len(iterable), n):
            yield iterable[i:i+n]
    else:
        l = []
        for x in iterable:
            l.append(x)
            if len(l) >= n:
                yield l
                l = []
        if l:
            yield l


@deconstructible
class FileUploadPath(object):
    def __init__(self, path, ext):
        self.path = path
        self.ext = ext

    def __call__(self, instance, filename):
        path = os.path.join(self.path, '%s.%s' % (uuid.uuid4(), self.ext))
        # insert a hash prefix in the beginning to distribute the load across S3 nodes
        # (https://aws.amazon.com/blogs/aws/amazon-s3-performance-tips-tricks-seattle-hiring-event/)
        path = os.path.join(hashlib.md5(path).hexdigest()[:4], path)
        return path


@deconstructible
class FileUploadPathKeepName(object):
    def __init__(self, path, ext):
        self.path = path
        self.ext = ext

    def __call__(self, instance, filename):
        path = os.path.join(self.path, '%s.%s' % (filename, self.ext))
        # insert a hash prefix in the beginning to distribute the load across S3 nodes
        # (https://aws.amazon.com/blogs/aws/amazon-s3-performance-tips-tricks-seattle-hiring-event/)
        path = os.path.join(hashlib.md5(path).hexdigest()[:4], path)
        return path


def ensuredir(dirpath):
    if os.path.exists(dirpath):
        return

    try:
        os.makedirs(dirpath)
    except OSError as exc:  # Python >2.5
        import errno
        if exc.errno == errno.EEXIST and os.path.isdir(dirpath):
            pass
        else:
            raise


def safe_save_content(temp_dir, filepath, content):
    # Download to a temp location, so there is not collision if multiple
    # processes are trying to download the same file
    f_temp = NamedTemporaryFile(dir=temp_dir, delete=False)
    f_temp.write(content)
    f_temp.close()
    # Move it to the final destination in an atomic operation
    os.rename(f_temp.name, filepath)


class EmptyModelBase(models.Model):
    """ Base class of all models, with no fields """

    def get_entry_dict(self):
        return {'id': self.id}

    def get_entry_attr(self):
        ct = ContentType.objects.get_for_model(self)
        return 'data-model="%s/%s" data-id="%s"' % (
            ct.app_label, ct.model, self.id)

    def get_entry_id(self):
        ct = ContentType.objects.get_for_model(self)
        return '%s.%s.%s' % (ct.app_label, ct.model, self.id)

    class Meta:
        abstract = True
        ordering = ['-id']


class ModelBase(EmptyModelBase):
    """ Base class of all models, with an 'added' field """
    added = models.DateTimeField(default=now)

    class Meta:
        abstract = True
        ordering = ['-id']


def get_opensurfaces_storage():
    return import_attribute(settings.OPENSURFACES_FILE_STORAGE)()


def import_function(module_function_str):
    tokens = module_function_str.split('.')
    module_name = '.'.join(tokens[:-1])
    func_name = tokens[-1]
    imported_module = __import__(
        module_name,
        globals=globals(),
        locals=locals(),
        fromlist=[func_name],
        level=-1,
    )
    return getattr(imported_module, func_name)


def resize_mindim(img, mindim):
    '''
    Resizes a PIL image so that the minimum dimension becomes mindim
    '''
    from pilkit.processors import ResizeToFit
    w, h = img.size
    if h < w:
        return ResizeToFit(height=mindim, upscale=True).process(img)
    else:
        return ResizeToFit(width=mindim, upscale=True).process(img)


def make_prefix_dirs(root_path, filename, levels=3):
    chunks = [root_path]
    chunks += [filename[i] for i in range(levels)]

    return os.path.join(*chunks)
