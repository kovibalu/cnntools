import os

from django.conf import settings
from django.core.management.base import BaseCommand

from cnntools.models import CaffeCNN, CaffeCNNTrainingRun
from cnntools.snapshot_utils import upload_snapshot
from cnntools.utils import create_default_trrun


class Command(BaseCommand):
    args = '<netid> <trainingrun_id> <iteration> <weightfile_relpath>'
    help = ('Adds a snapshot to the database. If trainingrun_id equals 0, we'
            ' create a dummy trainingrun to accommodate the snapshot.')

    def handle(self, *args, **option):
        if len(args) != 4:
            print 'Incorrect number of arguments!'
            print 'Usage: ./manage.py cnntools_add_snapshot [options] %s' % Command.args
            return

        netid = args[0]
        trainingrun_id = int(args[1])
        iteration = int(args[2])
        weightfile_relpath = args[3]
        weightfile_path = os.path.join(settings.CAFFE_ROOT, weightfile_relpath)

        if not CaffeCNN.objects.filter(netid=netid).exists():
            raise ValueError(
                'Couldn\'t find a CaffeCNN with the specified netid: {}'.format(netid)
            )
        caffe_cnn = CaffeCNN.objects.get(netid=netid)
        model_file_content = caffe_cnn.get_model_file_content()
        solver_file_content = caffe_cnn.get_solver_file_content()
        deploy_file_content = caffe_cnn.get_deploy_file_content()

        if not trainingrun_id:
            # Create a dummy trainingrun for the snapshot
            trrun = create_default_trrun(
                caffe_cnn,
                model_file_content,
                solver_file_content,
                deploy_file_content,
                100000,
            )
            trainingrun_id = trrun.id
        else:
            if not CaffeCNNTrainingRun.objects.filter(id=trainingrun_id).exists():
                raise ValueError(
                    'Couldn\'t find a CaffeCNNTrainingRun with the specified ID: {}'.format(trainingrun_id)
                )
            tr_netid = CaffeCNNTrainingRun.objects.filter(id=trainingrun_id).first().net.netid
            if tr_netid != netid:
                raise ValueError(
                    'The netid corresponding to the CaffeCNNTrainingRun doesn\'t match the provided netid: {} != {}:'.format(tr_netid, netid)
                )

        upload_snapshot(trainingrun_id, weightfile_path, iteration)
