import timeit
from django.conf import settings


class Timer:
    def __init__(self, message='Execution'):
        self.message = message

    def __enter__(self):
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args):
        self.end = timeit.default_timer()
        self.interval = self.end - self.start

        if hasattr(settings, 'SKIP_TIMER') and not settings.SKIP_TIMER:
            print '{} took {:.3f} seconds'.format(self.message, self.interval)
