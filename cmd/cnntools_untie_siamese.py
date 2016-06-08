import os

from django.conf import settings
from django.core.management.base import BaseCommand

from cnntools.utils import add_bkovacs_to_path, add_caffe_to_path


class Command(BaseCommand):
    args = '<trainingfile_source_relpath> <weights_source_relpath> <trainingfile_target_relpath> <weights_target_relpath>'
    help = 'Copies weights from one branch of the siamese network to the other, so it can be trained with untied weights'

    def handle(self, *args, **option):
        if len(args) != 5:
            print 'Incorrect number of arguments!'
            print 'Usage: ./manage.py cnntools_untie_siamese [options] %s' % Command.args
            return

        trainingfile_source_relpath = args[0]
        weights_source_relpath = args[1]
        trainingfile_target_relpath = args[2]
        weights_target_relpath = args[3]
        suffix = args[4]

        trainingfile_source_path = os.path.join(settings.CAFFE_ROOT, trainingfile_source_relpath)
        trainingfile_target_path = os.path.join(settings.CAFFE_ROOT, trainingfile_target_relpath)

        add_caffe_to_path()
        add_bkovacs_to_path()
        import caffe

        # Change working directory to Caffe
        os.chdir(settings.CAFFE_ROOT)

        net_source = caffe.Net(
            trainingfile_source_path,
            os.path.join(settings.CAFFE_ROOT, weights_source_relpath),
            caffe.TEST
        )
        net_target = caffe.Net(
            trainingfile_target_path,
            caffe.TEST
        )
        # Get all source names
        names = net_target.params.keys()

        print 'Computing param mapping...'
        param_mapping = {}
        for name in names:
            if suffix in name:
                orig_name = name.replace(suffix, '')
                if orig_name not in names:
                    print 'Warning: the corresponding name to {} ({}) is not among the blob names, skipping weight transfer!'
                else:
                    param_mapping[name] = orig_name

        for i, (t, s) in enumerate(param_mapping.iteritems()):
            if s not in net_source.params:
                print 'Couldn\'nt find {} among the source net params, skipping...'.format(s)
                continue

            # Weights and biases
            for blob_idx in (0, 1):
                print '%s %s %s <-- %s %s %s' % (
                    t, blob_idx, net_target.params[t][blob_idx].data.shape,
                    s, blob_idx, net_source.params[s][blob_idx].data.shape,
                )

                net_target.params[s][blob_idx].data[...] = net_source.params[s][blob_idx].data
                net_target.params[t][blob_idx].data[...] = net_source.params[s][blob_idx].data

        net_target.save(
            os.path.join(settings.CAFFE_ROOT, weights_target_relpath),
        )

        print 'Done.'

