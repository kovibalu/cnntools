import json
from datetime import timedelta

from cnntools.common_utils import (FileUploadPath, FileUploadPathKeepName,
                                   ModelBase, get_opensurfaces_storage)
from cnntools.utils import (gen_net_graph_svg, get_file_content,
                            get_svgs_from_output)
from django.db import models
# Receive the pre_delete signal and delete the file associated with the model instance.
from django.db.models.signals import pre_delete
from django.dispatch.dispatcher import receiver
from django.utils.timezone import now

STORAGE = get_opensurfaces_storage()


class CaffeCNN(ModelBase):
    """ A Caffe CNN model for training """

    #: human-friendly description
    description = models.TextField()

    #: human-friendly short description which can be shown as a tooltip. This
    #: can contain html too, for example a formula which describes how the
    #: network works.
    html_short_description = models.TextField(default='')

    #: network name
    netid = models.CharField(max_length=64, unique=True)

    FILE_OPTS = {
        'null': True,
        'blank': True,
        'max_length': 255,
        'storage': STORAGE,
    }

    #: Prototxt file which defines the model
    model_file = models.FileField(
        upload_to=FileUploadPath('cnntools', ext='prototxt'),
        **FILE_OPTS
    )

    #: Prototxt file which defines the default solver parameters
    solver_file = models.FileField(
        upload_to=FileUploadPath('cnntools', ext='prototxt'),
        **FILE_OPTS
    )

    #: Prototxt file which defines the deployed model
    deploy_file = models.FileField(
        upload_to=FileUploadPath('cnntools', ext='prototxt'),
        **FILE_OPTS
    )

    def get_model_file_content(self):
        return get_file_content(self.model_file)

    def get_solver_file_content(self):
        return get_file_content(self.solver_file)

    def get_deploy_file_content(self):
        return get_file_content(self.deploy_file)

    def get_net_graph_svg(self):
        return gen_net_graph_svg(self.get_model_file_content())

    def __unicode__(self):
        return "netid: %s" % (self.netid)


class CaffeCNNTrainingRun(ModelBase):
    #: The corresponding CNN model
    net = models.ForeignKey(CaffeCNN, related_name='training_runs', on_delete=models.PROTECT)

    #: How far did the network train
    final_iteration = models.IntegerField()

    #: How far we wanted to train
    max_iteration = models.IntegerField()

    FILE_OPTS = {
        'null': True,
        'blank': True,
        'max_length': 255,
        'storage': STORAGE,
    }

    #: Prototxt file which defines the model
    model_file_snapshot = models.FileField(
        upload_to=FileUploadPath('cnntools', ext='prototxt'),
        **FILE_OPTS
    )

    #: Prototxt file which defines the default solver parameters
    solver_file_snapshot = models.FileField(
        upload_to=FileUploadPath('cnntools', ext='prototxt'),
        **FILE_OPTS
    )

    #: Prototxt file which defines the deployed model
    deploy_file_snapshot = models.FileField(
        upload_to=FileUploadPath('cnntools', ext='prototxt'),
        **FILE_OPTS
    )

    #: JSON serialized output structure parsed from the Caffe training output
    outputs_json = models.TextField()

    #: JSON serialized output names parsed from the Caffe training output
    output_names_json = models.TextField()

    #: Last time this run was updated
    modified = models.DateTimeField(auto_now=True)

    # human-friendly description
    description = models.TextField(null=True, blank=True)

    def is_finished(self):
        # It's finished if there was no update in 10 hours
        return self.modified < now() - timedelta(hours=10) or \
            (self.max_iteration != 0 and self.final_iteration == self.max_iteration)

    def best_val_value(self, output_name, use_max=True):
        outputs, output_names = self.get_outputs()
        # We are only interested in the validation results
        outputs = outputs[1]
        output_names = output_names[1]
        # Search for output number by output name
        # Note: we choose the first one with the specified name, beware of name
        # collision!
        for o_num, o_name in output_names.iteritems():
            if o_name == output_name:
                output_num = o_num
                break

        op = outputs[output_num]
        if not op:
            return None

        # Find iteration corresponding to the highest/lowest value
        itnum_value_pairs = sorted(op.items(), key=lambda x: x[1], reverse=use_max)
        return itnum_value_pairs[0]

    @property
    def runtime(self):
        return self.modified - self.added

    @property
    def final_shapshot(self):
        return self.snapshots.order_by('-id')[0]

    @property
    def est_rem_time(self):
        if self.final_iteration == 0:
            return timedelta(minutes=0)

        ratio = float(self.max_iteration - self.final_iteration) / self.final_iteration
        t_seconds = self.runtime.total_seconds() * ratio
        return timedelta(seconds=t_seconds)

    def get_outputs(self):
        outputs = json.loads(self.outputs_json)
        output_names = json.loads(self.output_names_json)

        return outputs, output_names

    def get_svgs(self, simplified=False):
        outputs, output_names = self.get_outputs()
        return get_svgs_from_output(
            outputs, output_names, simplified=simplified
        )

    def get_model_file_content(self):
        return get_file_content(self.model_file_snapshot)

    def get_solver_file_content(self):
        return get_file_content(self.solver_file_snapshot)

    def get_deploy_file_content(self):
        return get_file_content(self.deploy_file_snapshot)

    def get_net_graph_svg(self):
        return gen_net_graph_svg(self.get_model_file_content())


@receiver(pre_delete, sender=CaffeCNN)
def CaffeCNN_delete(sender, instance, **kwargs):
    # Pass false so FileField doesn't save the model.
    if instance.model_file:
        instance.model_file.delete(save=False)

    if instance.solver_file:
        instance.solver_file.delete(save=False)

    if instance.deploy_file:
        instance.deploy_file.delete(save=False)


@receiver(pre_delete, sender=CaffeCNNTrainingRun)
def CaffeCNNTrainingRun_delete(sender, instance, **kwargs):
    # Pass false so FileField doesn't save the model.
    if instance.model_file_snapshot:
        instance.model_file_snapshot.delete(save=False)

    if instance.solver_file_snapshot:
        instance.solver_file_snapshot.delete(save=False)

    if instance.deploy_file_snapshot:
        instance.deploy_file_snapshot.delete(save=False)


class CaffeCNNSnapshot(ModelBase):
    #: The corresponding CNN training run
    training_run = models.ForeignKey(CaffeCNNTrainingRun, related_name='snapshots', on_delete=models.PROTECT)

    #: The iteration number when we took this snapshot
    iteration = models.IntegerField()

    #: hash for detecting incorrect uploads/downloads
    sha1 = models.CharField(max_length=40)

    FILE_OPTS = {
        'null': True,
        'blank': True,
        'max_length': 255,
        'storage': STORAGE,
    }

    #: Prototxt file which defines the model
    model_snapshot = models.FileField(
        upload_to=FileUploadPathKeepName('cnntools/snapshots', ext='caffemodel'),
        **FILE_OPTS
    )


@receiver(pre_delete, sender=CaffeCNNSnapshot)
def CaffeCNNSnapshot_delete(sender, instance, **kwargs):
    # Pass false so FileField doesn't save the model.
    if instance.model_snapshot:
        instance.model_snapshot.delete(save=False)
