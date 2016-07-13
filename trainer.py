import json
import os
import time

from django.conf import settings

from cnntools import caffefileproc
from cnntools.common_utils import ensuredir
from cnntools.models import CaffeCNNTrainingRun
from cnntools.snapshot_utils import upload_snapshot
from cnntools.utils import (add_caffe_to_path, add_to_path,
                            random_file_from_content)


def _load_training_run(caffe_cnn_trrun_id):
    entry = CaffeCNNTrainingRun.objects.get(id=caffe_cnn_trrun_id)
    outputs_str = json.loads(entry.outputs_json)
    # Convert to numbers
    outputs = []
    for op_str in outputs_str:
        op = {}
        for op_num_str, its_str in op_str.iteritems():
            its = {}
            for it_str, val_str in its_str.iteritems():
                its[int(it_str)] = float(val_str)
            op[int(op_num_str)] = its
        outputs.append(op)

    max_iter = entry.max_iteration

    return outputs, max_iter


def _refresh_training_run(caffe_cnn_trrun_id, outputs, output_names, itnum,
                          max_iter):
    outputs_json = json.dumps(outputs)
    output_names_json = json.dumps(output_names)
    entry = CaffeCNNTrainingRun.objects.get(id=caffe_cnn_trrun_id)
    entry.outputs_json = outputs_json
    entry.output_names_json = output_names_json
    entry.final_iteration = itnum
    entry.max_iteration = max_iter
    entry.save()


def _refresh_solverfile(caffe_cnn_trrun_id, solver_file_content):
    entry = CaffeCNNTrainingRun.objects.get(id=caffe_cnn_trrun_id)
    # Overwrite solverfile content with new data
    entry.solver_file_snapshot.delete(save=False)
    entry.solver_file_snapshot = random_file_from_content(solver_file_content)
    entry.save()


def extract_batchsize_testsetsize(model_file_content):
    from caffe.proto import caffe_pb2

    # Note that we handle only MULTI_IMAGE_DATA layer and some other layers (see below), we require include.phase be TEST
    model_params = caffefileproc.parse_model_definition_file_content(model_file_content)
    batch_size = None
    testset_size = None
    layer_types = [
        'ImageData',
        'MultiImageData',
        'Python',
    ]
    for layer in model_params.layer:
        if layer.type in layer_types:
            if not layer.include:
                continue
            if layer.include[0].phase == caffe_pb2.TEST:
                if layer.type == 'Python':
                    # Try to parse json
                    try:
                        params = json.loads(layer.python_param.param_str)
                        batch_size = params['batch_size']
                        source = params['source']
                    except Exception as e:
                        print 'Failed to parse param_str: ', e
                        print 'Skipping this layer.'
                        continue
                elif layer.type == 'ImageData':
                    batch_size = layer.image_data_param.batch_size
                    source = layer.image_data_param.source
                elif layer.type == 'MultiImageData':
                    batch_size = layer.multi_image_data_param.batch_size
                    source = layer.multi_image_data_param.source
                else:
                    continue

                # Note that the source should be relative to the caffe root path!
                testset_size = len(caffefileproc.freadlines(os.path.join(settings.CAFFE_ROOT, source)))
                break

    return batch_size, testset_size


def setup_solverfile(model_name, model_file_content, solver_file_content,
                     options, device_id):
    rand_name = str(time.time())
    root_path = os.path.join(
        settings.CAFFE_ROOT,
        'training_runs',
        '-'.join([rand_name, model_name])
    )

    ensuredir(root_path)
    trainfilename = 'train_val.prototxt'
    solverfilename = 'solver.prototxt'
    trainfile_path = os.path.join(root_path, trainfilename)
    solverfile_path = os.path.join(root_path, solverfilename)

    # Save the model_file_content to a file, so Caffe can read it
    with open(trainfile_path, 'w') as f:
        f.write(model_file_content)

    # copy the sample solver file and modify it
    solver_params = caffefileproc.parse_solver_file_content(solver_file_content)

    # modify solver params according to the command line parameters
    solver_params.net = trainfile_path
    if 'base_lr' in options and options['base_lr'] is not None:
        solver_params.base_lr = options['base_lr']

    # Switch on debug_info to facilitate debugging
    if 'debug_info' in options and options['debug_info'] is not None:
        solver_params.debug_info = options['debug_info']

    snapshot_path = os.path.join(root_path, 'snapshots')
    ensuredir(snapshot_path)
    solver_params.snapshot_prefix = os.path.join(
        snapshot_path,
        'train_{}-base_lr{}'.format(model_name, solver_params.base_lr)
    )

    # compute the proper test_iter
    batch_size, testset_size = extract_batchsize_testsetsize(model_file_content)

    if batch_size and testset_size:
        if options['verbose']:
            print 'Extracted batch_size ({0}) and testset_size ({1})'.format(
                batch_size, testset_size)
        # Note the solver file should have exactly one test_iter
        solver_params.test_iter[0] = int(testset_size/batch_size)
    else:
        if options['verbose']:
            print 'WARNING: Couldn\'t find the batch_size or the source file ' + \
                'containing the testset, please set the test_iter to ' + \
                'testset_size / batch_size!'

    # Setting random seed value for reproducible results
    solver_params.random_seed = settings.CAFFE_SEED

    add_caffe_to_path()
    import caffe
    from caffe.proto import caffe_pb2

    if settings.CAFFE_GPU and (options['cpu'] is None or not options['cpu']):
        if options['verbose']:
            print 'Using GPU'
        caffe.set_mode_gpu()
        caffe.set_device(device_id)
        solver_params.solver_mode = caffe_pb2.SolverParameter.GPU
    else:
        if options['verbose']:
            print 'Using CPU'
        caffe.set_mode_cpu()
        solver_params.solver_mode = caffe_pb2.SolverParameter.CPU

    caffefileproc.save_protobuf_file(solverfile_path, solver_params)

    return solver_params, solverfile_path


def train_network(solver_params, solverfile_path, options,
                  caffe_cnn_trrun_id):
    add_caffe_to_path()
    import caffe

    for p in settings.TRAINING_EXTRA_PYTHON_PATH:
        add_to_path(p)

    restore = False

    solver = caffe.SGDSolver(solverfile_path)
    if 'weights' in options and options['weights'] is not None:
        _, ext = os.path.splitext(options['weights'])
        if ext == '.solverstate':
            solver.restore(
                os.path.join(settings.CAFFE_ROOT, options['weights'])
            )
            restore = True
        else:
            solver.net.copy_from(
                os.path.join(settings.CAFFE_ROOT, options['weights'])
            )
    # If the data layer is python, we try to set the random seed for
    # reproducibility
    # Note: The python implementation of the layer should have a
    # "set_random_seed" function. If we can't find a function with this name,
    # we won't set the random seed
    data_layer = solver.net.layers[0]
    if data_layer.type == 'Python':
        set_random_seed = getattr(data_layer, 'set_random_seed', None)
        if callable(set_random_seed):
            set_random_seed(settings.CAFFE_SEED)

    # {key: output_num, value: output_name}
    output_names = [{}, {}]
    # for backward compatibility
    name_to_num = [{}, {}]
    # {key: output_num, value: {key: it_num value: output_value}}
    outputs = [{}, {}]

    for i, op in enumerate(solver.net.outputs):
        output_names[0][i] = op
        name_to_num[0][op] = i
        outputs[0][i] = {}

    op_num = 0
    for test_net in solver.test_nets:
        for op in test_net.outputs:
            output_names[1][op_num] = op
            name_to_num[1][op] = op_num
            outputs[1][op_num] = {}
            op_num += 1

    if restore:
        # Load back the saved outputs, so we can start from the figures where
        # we left off
        outputs, max_iter = _load_training_run(caffe_cnn_trrun_id)
        start_it = int(solver.iter)
        # Filter out data which happened after the snapshot
        for op in outputs:
            for op_num, its in op.iteritems():
                to_remove = []
                for it in its:
                    if it > start_it:
                        to_remove.append(it)
                for it in to_remove:
                    its.pop(it)
    else:
        max_iter = solver_params.max_iter
        start_it = 0

    final_snapshot_id = None
    for it in range(start_it, max_iter+1):
        start = time.clock()
        solver.step(1)  # SGD by Caffe
        elapsed = time.clock() - start
        if options['verbose']:
            print 'One iteration took {:.2f} seconds'.format(elapsed)

        display = solver_params.display and it % solver_params.display == 0
        if display:
            for op in solver.net.outputs:
                val = solver.net.blobs[op].data
                op_num = name_to_num[0][op]
                outputs[0][op_num][it] = float(val)

        test_display = solver_params.test_interval and \
            it % solver_params.test_interval == 0 and \
            (it != 0 or solver_params.test_initialization)
        if test_display:
            for i, test_net in enumerate(solver.test_nets):
                for op in test_net.outputs:
                    val = solver.test_mean_scores[i][op]
                    op_num = name_to_num[1][op]
                    outputs[1][op_num][it] = float(val)

        if display or test_display:
            _refresh_training_run(
                caffe_cnn_trrun_id,
                outputs,
                output_names,
                it,
                max_iter,
            )

        snapshot = it % solver_params.snapshot == 0 and it != 0
        if snapshot:
            snapshot_path = os.path.join(
                settings.CAFFE_ROOT,
                '{}_iter_{}.caffemodel'.format(
                    solver_params.snapshot_prefix,
                    it
                )
            )
            final_snapshot = upload_snapshot(
                caffe_cnn_trrun_id=caffe_cnn_trrun_id,
                snapshot_path=snapshot_path,
                it=it,
                verbose=options['verbose'],
            )
            final_snapshot_id = final_snapshot.id

    return final_snapshot_id


def start_training(model_name, model_file_content, solver_file_content,
                   options, caffe_cnn_trrun_id, device_id=0):
    if options['verbose']:
        print 'Running training for model {}...'.format(model_name)
        print 'with options: {}'.format(options)

    solver_params, solverfile_path = setup_solverfile(
        model_name, model_file_content, solver_file_content, options, device_id
    )

    # Save the final solver file's content to database
    _refresh_solverfile(
        caffe_cnn_trrun_id,
        caffefileproc.gen_protobuf_file_content(solver_params),
    )

    # Change working directory to Caffe
    os.chdir(settings.CAFFE_ROOT)
    # TODO: Figure out to detect divergence and that the loss doesn't decrease,
    # so we can stop training
    return train_network(
        solver_params, solverfile_path, options, caffe_cnn_trrun_id,
    )
