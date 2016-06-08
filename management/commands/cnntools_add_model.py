import os

from cnntools.models import CaffeCNN
from cnntools.utils import random_file_from_content
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    args = '<netid> <modelfile_relpath> <solverfile_relpath> <deployfile_relpath> <description>'
    help = 'Adds a CaffeCNN model to the database'

    def handle(self, *args, **option):
        if len(args) != 5:
            print 'Incorrect number of arguments!'
            print 'Usage: ./manage.py cnntools_add_model [options] %s' % Command.args
            return

        netid = args[0]
        modelfile_path = args[1]
        solverfile_path = args[2]
        deployfile_path = args[3]
        description = args[4]

        # If the files are already there, delete them first
        if CaffeCNN.objects.filter(netid=netid).exists():
            caffe_cnn = CaffeCNN.objects.get(netid=netid)
            caffe_cnn.model_file.delete()
            caffe_cnn.solver_file.delete()
            caffe_cnn.deploy_file.delete()
            caffe_cnn.save()

        CaffeCNN.objects.update_or_create(
            netid=netid,
            defaults={
                'description': description,
                'model_file': random_file_from_content(open(modelfile_path).read()),
                'solver_file': random_file_from_content(open(solverfile_path).read()),
                'deploy_file': random_file_from_content(open(deployfile_path).read()),
            }
        )

