import json
import os
from optparse import make_option

from cnntools.models import CaffeCNN
from cnntools.tasks import schedule_training
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    args = '<netid> <local?> <base_lr?> <weights?> <cpu?> <debug_info?> <desc?>'
    help = 'Starts training a CaffeCNN model'

    option_list = BaseCommand.option_list + (
        make_option(
            '--local',
            action='store_true',
            dest='local',
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
            '--weight_decay',
            action='store',
            type='float',
            dest='weight_decay',
            help='Weight decay',
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
        make_option(
            '--desc',
            action='store',
            type='string',
            dest='description',
            default='No description.',
            help='Description of the training run (what is different?)',
        ),
        make_option(
            '--noverbose',
            action='store_false',
            dest='verbose',
            help='If specified, we will not print additional information',
        ),
        make_option(
            '--data_layer_params',
            action='store',
            type='string',
            dest='data_layer_params',
            default='{}',
            help='Additional parameters passed to the python data layer',
        ),
        make_option(
            '--start_snapshot',
            action='store_true',
            dest='start_snapshot',
            help='If specified, we will save a snapshot at before starting training',
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

        if 'data_layer_params' in options:
            options['data_layer_params'] = json.loads(options['data_layer_params'])

        schedule_training(
            netid=netid,
            options=options,
        )

