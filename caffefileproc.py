from datetime import datetime

from cnntools.utils import add_caffe_to_path
# Make sure that caffe is on the python path:
from google import protobuf


def freadlines(filepath, strip=True):
    with open(filepath, 'r') as f:
        if strip:
            lines = [s.strip() for s in f.readlines()]
        else:
            lines = f.readlines()

    return lines


def fwritelines(filepath, lines, endline=True):
    with open(filepath, 'w') as f:
        for l in lines:
            if endline:
                l = '{0}\n'.format(l)

            f.write(l)


def parse_solver_file(filepath):
    '''
    Returns a protobuf object with the solver parameters
    '''
    lines = freadlines(filepath)
    return parse_solver_file_content('\n'.join(lines))


def parse_solver_file_content(file_content):
    '''
    Returns a protobuf object with the solver parameters
    '''
    add_caffe_to_path()
    from caffe.proto import caffe_pb2
    solver_params = caffe_pb2.SolverParameter()
    protobuf.text_format.Merge(file_content, solver_params)

    return solver_params


def gen_protobuf_file_content(protobuf_params):
    return protobuf.text_format.MessageToString(protobuf_params)


def save_protobuf_file(filepath, protobuf_params):
    '''
    Saves a protobuf file using the specified parameters
    '''
    with open(filepath, 'w') as f:
        f.write('# GENERATED BY TRAINER.PY {0}\n'.format(datetime.now()))
        f.write(gen_protobuf_file_content(protobuf_params))


def parse_model_definition_file(filepath):
    '''
    Parse protobuf file
    '''
    lines = freadlines(filepath)
    return parse_model_definition_file_content('\n'.join(lines))


def parse_model_definition_file_content(file_content):
    '''
    Parse protobuf file content
    '''
    add_caffe_to_path()
    from caffe.proto import caffe_pb2
    model_params = caffe_pb2.NetParameter()
    protobuf.text_format.Merge(file_content, model_params)

    return model_params

