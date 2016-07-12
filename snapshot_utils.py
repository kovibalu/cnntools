import hashlib
import os
from tempfile import NamedTemporaryFile

import numpy as np

from cnntools.caffefileproc import parse_model_definition_file
from cnntools.common_utils import ensuredir, safe_save_content
from cnntools.models import CaffeCNNSnapshot, CaffeCNNTrainingRun
from cnntools.utils import (add_caffe_to_path, get_file_content,
                            named_file_from_content)
from django.conf import settings


def upload_snapshot(caffe_cnn_trrun_id, snapshot_path, it):
    print 'Uploading snapshot {}...'.format(snapshot_path)
    snapshot_content = open(snapshot_path).read()
    sha1 = hashlib.sha1(snapshot_content).hexdigest()

    training_run = CaffeCNNTrainingRun.objects.get(id=caffe_cnn_trrun_id)
    # Training run ID + iteration number uniquely identifies the snapshot
    name = '{}-trrun{}-it{}'.format(
        training_run.net.netid,
        caffe_cnn_trrun_id,
        it,
    )

    return CaffeCNNSnapshot.objects.create(
        training_run_id=caffe_cnn_trrun_id,
        iteration=it,
        sha1=sha1,
        model_snapshot=named_file_from_content(snapshot_content, name)
    )


def download_snapshot(snapshot_id, transfer):
    snapshot = CaffeCNNSnapshot.objects.get(id=snapshot_id)
    snapshot_dir = os.path.join(
        'snapshots',
        str(snapshot_id),
    )
    snapshot_dirpath = os.path.join(settings.CAFFE_ROOT, snapshot_dir)
    ensuredir(snapshot_dirpath)
    weights_relpath = os.path.join(
        snapshot_dir,
        'snapshot.caffemodel'
    )
    deployfile_relpath = os.path.join(
        snapshot_dir,
        'deploy.prototxt'
    )
    trainfile_relpath = os.path.join(
        snapshot_dir,
        'train_val.prototxt'
    )
    weights_path = os.path.join(settings.CAFFE_ROOT, weights_relpath)
    if not os.path.exists(weights_path):
        print 'Downloading snapshot {}...'.format(weights_path)
        snapshot_content = get_file_content(snapshot.model_snapshot)
        sha1 = hashlib.sha1(snapshot_content).hexdigest()

        if sha1 != snapshot.sha1:
            raise ValueError(
                'The SHA1 digest of the downloaded snapshot '
                'doesn\'t match the uploaded one'
            )
        safe_save_content(snapshot_dirpath, weights_path, snapshot_content)

    deployfile_path = os.path.join(settings.CAFFE_ROOT, deployfile_relpath)
    if not os.path.exists(deployfile_path):
        safe_save_content(
            snapshot_dirpath, deployfile_path,
            snapshot.training_run.get_deploy_file_content()
        )

    trainfile_path = os.path.join(settings.CAFFE_ROOT, trainfile_relpath)
    if not os.path.exists(trainfile_path):
        safe_save_content(
            snapshot_dirpath, trainfile_path,
            snapshot.training_run.get_model_file_content()
        )

    if transfer:
        # Transfer the weights so they are compatible with the fully
        # convolutional network
        transferred_weights_relpath = os.path.join(
            snapshot_dir,
            'snapshot-conv.caffemodel'
        )
        if not os.path.exists(os.path.join(settings.CAFFE_ROOT, transferred_weights_relpath)):
            transfer_weights(
                temp_dir=snapshot_dirpath,
                deployfile_source_path=os.path.join(settings.CAFFE_ROOT, trainfile_relpath),
                weights_source_path=os.path.join(settings.CAFFE_ROOT, weights_relpath),
                deployfile_target_path=os.path.join(settings.CAFFE_ROOT, deployfile_relpath),
                weights_target_path=os.path.join(settings.CAFFE_ROOT, transferred_weights_relpath),
                verbose=True,
            )
    else:
        transferred_weights_relpath = weights_relpath

    return deployfile_relpath, transferred_weights_relpath


def get_fc_layers(deployfile_source):
    layers = deployfile_source.layer
    fc_type = 'InnerProduct'

    return [
        layer.name
        for layer in layers
        if layer.type == fc_type
    ]


def convolutionize_net(deployfile_source):
    fc_layers = get_fc_layers(deployfile_source)
    print 'Found fully connected layers:', fc_layers

    from caffe.proto import caffe_pb2
    deployfile_target = caffe_pb2.NetParameter()
    deployfile_target.CopyFrom(deployfile_source)
    deployfile_target.ClearField('layer')

    param_mapping = {}

    for layer in deployfile_source.layer:
        layer_target = deployfile_target.layer.add()
        layer_target.CopyFrom(layer)
        if layer.name in fc_layers:
            layer_target.name = layer_target.name + '-conv'
            layer_target.type = 'Convolution'
            param_mapping[layer_target.name] = layer.name
            # TODO: Compute proper convolution size....

    return deployfile_target, param_mapping


def print_net_weight_stats_file(deployfile_path, weights_path):
    add_caffe_to_path()
    import caffe
    net = caffe.Net(deployfile_path, weights_path, caffe.TEST)
    print_net_weight_stats(net)


def print_net_weight_stats(net):
    print 'Net weights stats:'
    for layer_name, weights in net.params.iteritems():
        print layer_name
        print '\tweights mean: %f +/- %f' % (np.mean(weights[0].data), np.std(weights[0].data))
        print '\tbias mean: %f +/- %f' % (np.mean(weights[1].data), np.std(weights[1].data))


def transfer_weights(temp_dir, deployfile_source_path, weights_source_path,
                     deployfile_target_path, weights_target_path,
                     verbose=True):

    deployfile_source = parse_model_definition_file(deployfile_source_path)

    # Modify deploy file
    deployfile_target, param_mapping = convolutionize_net(deployfile_source)
    #save_protobuf_file(deployfile_target_path, deployfile_target)

    add_caffe_to_path()
    import caffe
    net_source = caffe.Net(
        deployfile_source_path,
        weights_source_path,
        caffe.TEST
    )
    net_target = caffe.Net(
        deployfile_target_path,
        weights_source_path,
        caffe.TEST
    )

    for t, s in param_mapping.iteritems():
        if t not in net_target.params:
            print 'WARNING: Couldn\'t find "%s" layer in the target model definition file, skipping...' % t
            continue

        for blob_idx in (0, 1):
            if verbose:
                print '%s %s %s <-- %s %s %s' % (
                    t, blob_idx, net_target.params[t][blob_idx].data.shape,
                    s, blob_idx, net_source.params[s][blob_idx].data.shape,
                )
            net_target.params[t][blob_idx].data[...] = (
                np.reshape(
                    net_source.params[s][blob_idx].data,
                    net_target.params[t][blob_idx].data.shape
                )
            )

    if verbose:
        print_net_weight_stats(net_target)

    # Download to a temp location, so there is not collision if multiple
    # processes are trying to download the same file
    f_temp = NamedTemporaryFile(dir=temp_dir, delete=False)
    f_temp.close()

    # Use the temp file's name
    net_target.save(f_temp.name)
    # Move it to the final destination in an atomic operation
    os.rename(f_temp.name, weights_target_path)


def load_net_from_snapshot(snapshot_id):
    deployfile_relpath, weights_relpath = download_snapshot(
        snapshot_id=snapshot_id,
        transfer=False,
    )

    add_caffe_to_path()
    import caffe
    deployfile_path = os.path.join(settings.CAFFE_ROOT, deployfile_relpath)
    weights_path = os.path.join(settings.CAFFE_ROOT, weights_relpath)

    return caffe.Net(deployfile_path, weights_path, caffe.TEST)
