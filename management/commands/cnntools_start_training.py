import os
from optparse import make_option

from cnntools.models import CaffeCNN
from cnntools.tasks import start_training_task
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    args = '<netid> <local?> <base_lr?> <weights?> <cpu?> <debug_info?>'
    help = 'Starts training a CaffeCNN model'

    option_list = BaseCommand.option_list + (
        make_option(
            '--local',
            action='store_true',
            dest='local',
            default=False,
            help='Start the training locally, instead of dispatching to Celery.',
        ),
        make_option(
            '--base_lr',
            action='store',
            type='float',
            dest='base_lr',
            help='Base learning rate to start with',
        ),
        make_option(
            '--weights',
            action='store',
            type='string',
            dest='weights',
            help='Relative path to the weights to finetune from or solverstate to continue from',
        ),
        make_option(
            '--cpu',
            action='store_true',
            dest='cpu',
            help='Use CPU for training instead of GPU',
        ),
        make_option(
            '--debug_info',
            action='store_true',
            dest='debug_info',
            help='Show debug info while training',
        ),
        make_option(
            '--caffe_cnn_trrun_id',
            action='store',
            type='int',
            dest='caffe_cnn_trrun_id',
            help='This should be specified if we want to continue training a model!',
        ),
    )

    def handle(self, *args, **options):
        if len(args) != 1:
            print 'Incorrect number of arguments!'
            print 'Usage: ./manage.py cnntools_start_training [options] %s' % Command.args
            return

        netid = args[0]

        if not CaffeCNN.objects.filter(netid=netid).exists():
            raise ValueError(
                'The specified netid doesn\'t exist: {}'.format(netid)
            )

        if 'caffe_cnn_trrun_id' not in options or options['caffe_cnn_trrun_id'] is None:
            if 'weights' in options and options['weights'] is not None:
                _, ext = os.path.splitext(options['weights'])
                if ext == '.solverstate':
                    raise ValueError((
                        '"caffe_cnn_trrun_id" must be specified if you '
                        'want to continue training from a solverstate!'
                    ))

        if options['local']:
            start_training_task(
                netid=netid,
                options=options,
            )
        else:
            start_training_task.delay(
                netid=netid,
                options=options,
            )

