from django.core.management.base import BaseCommand

from cnntools.snapshot_utils import transfer_weights


class Command(BaseCommand):
    args = '<deployfile_source_relpath> <weights_source_relpath> <deployfile_target_relpath> <weights_target_relpath>'
    help = 'Converts a net into a fully convolutional net by transforming all inner product layers to convolutional layers'

    def handle(self, *args, **option):
        if len(args) != 4:
            print 'Incorrect number of arguments!'
            print 'Usage: ./manage.py cnntools_convolutionalize_net [options] %s' % Command.args
            return

        deployfile_source_relpath = args[0]
        weights_source_relpath = args[1]
        deployfile_target_relpath = args[2]
        weights_target_relpath = args[3]

        transfer_weights(
            temp_dir=None,
            deployfile_source_relpath=deployfile_source_relpath,
            weights_source_relpath=weights_source_relpath,
            deployfile_target_relpath=deployfile_target_relpath,
            weights_target_relpath=weights_target_relpath,
            verbose=True,
        )

