import timeit
from django.conf import settings


class Timer:
    def __init__(self, message='Execution', force_show=False):
        self.message = message
        self.force_show = force_show

    def __enter__(self):
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args):
        self.end = timeit.default_timer()
        self.interval = self.end - self.start

        skip_timer = hasattr(settings, 'SKIP_TIMER') and settings.SKIP_TIMER
        if not skip_timer or self.force_show:
            print '{} took {:.3f} seconds'.format(self.message, self.interval)
