import json
import os
import random
import re
import StringIO
import sys
import uuid

import numpy as np

from django.conf import settings
from django.core.files import File


def sample_max_count(arr, max_count, seed=123):
    if len(arr) > max_count:
        random.seed(seed)
        return random.sample(arr, max_count)
    else:
        return arr


def get_file_content(file_handle):
    if not file_handle:
        return None

    file_handle.seek(0)
    return file_handle.read()


def named_file_from_content(content, name):
    buff = StringIO.StringIO()
    buff.write(content)
    buff.seek(0)
    return File(buff, name=name)


def random_file_from_content(content):
    return named_file_from_content(content, str(uuid.uuid4()))


def plot_2D_arrays(arrs, title='', xlabel='', xinterval=None, ylabel='', yinterval=None, line_names=[], simplified=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.clf()
    sns.set_style('darkgrid')

    for arr in arrs:
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                'The array should be 2D and the second dimension should be 2!'
                ' Shape: %s' % str(arr.shape)
            )

        plt.plot(arr[:, 0], arr[:, 1])

    # If simplified, we don't write text anywhere
    if not simplified:
        plt.title(title[:30])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if line_names:
            plt.legend(line_names, loc=6, bbox_to_anchor=(1, 0.5))

    if xinterval:
        plt.xlim(xinterval)
    if yinterval:
        plt.ylim(yinterval)

    plt.tight_layout()


def plot_and_svg_2D_arrays(arrs, xlabel='', xinterval=None, ylabel='', yinterval=None, line_names=[], simplified=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plot_2D_arrays(arrs, '', xlabel, xinterval, ylabel, yinterval, line_names, simplified)

    buf = StringIO.StringIO()
    plt.savefig(buf, format='svg', bbox_inches='tight')
    plt.clf()

    return buf.getvalue()


def plot_and_save_2D_arrays(filename, arrs, xlabel='', xinterval=None, ylabel='', yinterval=None, line_names=[], simplified=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    name, ext = os.path.splitext(os.path.basename(filename))
    plot_2D_arrays(arrs, name, xlabel, xinterval, ylabel, yinterval, line_names, simplified)
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()


def plot_and_save_2D_array(filename, arr, xlabel='', xinterval=None, ylabel='', yinterval=None, simplified=False):
    plot_and_save_2D_arrays(filename, [arr], xlabel, xinterval, ylabel, yinterval, line_names=[], simplified=simplified)


def create_figure_data(data_dict):
    ret = np.empty((len(data_dict), 2))
    keys = sorted(data_dict.keys(), key=lambda x: int(x))
    for idx, itnum in enumerate(keys):
        ret[idx, 0] = int(itnum)
        ret[idx, 1] = float(data_dict[itnum])

    return ret


def get_disp_config():
    disp_config = {}
    # loss
    disp_config['loss'] = {'name': 'Loss', 'yaxis_name': 'Loss', 'yinterval_policy': 'max'}
    # accuracy
    disp_config['accuracy'] = {'name': 'Accuracy', 'yaxis_name': 'Accuracy', 'yinterval_policy': 'fix', 'yinterval_value': [0, 1]}
    # other
    disp_config['other'] = {'name': 'Output', 'yaxis_name': 'Output'}

    return disp_config


def filter_disp_type(disp_config, output_name):
    for disp_type in disp_config:
        if disp_type in output_name.lower():
            return disp_type

    return 'other'


def plot_svg_figures(figure_arrs, line_names, dc, xlabel, simplified):
    '''Plots a bunch of figures with the specified display config (dc)'''

    # Choose different settings depending on the config
    if 'yinterval_policy' not in dc:
        yinterval = None
    elif dc['yinterval_policy'] == 'max':
        ymax = np.max([
            np.max(fa[:, 1])
            for fa in figure_arrs
            if fa.size > 0
        ])
        yinterval = [0, ymax*1.1]
    elif dc['yinterval_policy'] == 'fix':
        yinterval = dc['yinterval_value']
    else:
        yinterval = None

    return plot_and_svg_2D_arrays(
        figure_arrs,
        xlabel=xlabel,
        ylabel=dc['yaxis_name'], yinterval=yinterval,
        line_names=line_names,
        simplified=simplified,
    )


def get_svgs_from_output(outputs, output_names, simplified=False):
    '''
    Parses all outputs of one network and plots them aggregated by different
    types (loss, accuracy, other).
    '''
    disp_config = get_disp_config()

    # Partition outputs into 'loss', 'accuracy', 'other'
    # contains index pairs: 0/1 (training/test) and output_num
    indices = {disp_type: [] for disp_type in disp_config}

    # Train, test
    for i in range(2):
        for output_num, output_name in output_names[i].iteritems():
            disp_type = filter_disp_type(disp_config, output_name)
            indices[disp_type].append([i, output_num])

    svgs = {}
    # Collect data and draw figures
    for disp_type, dc in disp_config.iteritems():
        figure_arrs = []
        line_names = []
        line_template_str = ['Train {0}', 'Test {0}']

        for i, output_num in indices[disp_type]:
            op = outputs[i][output_num]
            on = output_names[i][output_num]

            figure_arrs.append(create_figure_data(op))
            line_names.append(line_template_str[i].format(on))

        if not figure_arrs:
            continue

        svgs[dc['name']] = plot_svg_figures(
            figure_arrs, line_names, dc, 'Iteration number', simplified
        )

    return svgs


def plot_svg_net_weights(weights_arr, title, xlabel, ylabel):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.clf()
    sns.set_style('darkgrid')

    plt.plot(weights_arr)

    plt.title(title[:30])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    buf = StringIO.StringIO()
    plt.savefig(buf, format='svg', bbox_inches='tight')
    plt.clf()

    return buf.getvalue()


def get_svgs_from_net(net):
    weight_plots = []
    for layer_name, weights in net.params.iteritems():
        for i, label in enumerate(['weights', 'biases']):
            if len(weights) <= i:
                continue

            title = '%s - %s' % (layer_name, label)
            svg = plot_svg_net_weights(
                weights_arr=np.ravel(weights[i].data),
                title=title,
                xlabel='Weight element index',
                ylabel='Weight value',
            )
            shape_text = ', '.join([
                str(x)
                for x in weights[i].data.shape
            ])
            weight_plots.append({
                'title': title,
                'svg': svg,
                'shape': shape_text,
            })

    return weight_plots


def add_to_path(p):
    if p not in sys.path:
        sys.path.insert(0, p)


def add_caffe_to_path():
    caffe_python_root = os.path.join(settings.CAFFE_ROOT, 'python')
    add_to_path(caffe_python_root)


def get_worker_name():
    from billiard import current_process
    p = current_process()
    return p.initargs[1].split('@')[1]


def get_worker_gpu_device_id():
    worker_name = get_worker_name()
    chunks = worker_name.split('#')

    devid_pattern = 'dev(\\d+)'
    # Default value
    device_id = 0

    # We choose the last chunk which matches the pattern
    for c in chunks:
        match = re.search(devid_pattern, c)
        if match:
            device_id = int(match.groups()[0])

    return device_id


def create_default_trrun(caffe_cnn, model_file_content, solver_file_content,
                         deploy_file_content, description='', final_iteration=0):
    from cnntools.models import CaffeCNNTrainingRun
    return CaffeCNNTrainingRun.objects.create(
        net=caffe_cnn,
        final_iteration=final_iteration,
        max_iteration=final_iteration,
        model_file_snapshot=random_file_from_content(model_file_content),
        solver_file_snapshot=random_file_from_content(solver_file_content),
        deploy_file_snapshot=random_file_from_content(deploy_file_content),
        description=description,
        outputs_json=json.dumps([{}, {}]),
        output_names_json=json.dumps([{}, {}]),
    )


def gen_net_graph_svg(model_file_content):
    add_caffe_to_path()
    from caffe.draw import draw_net
    from cnntools.caffefileproc import parse_model_definition_file_content
    model_params = parse_model_definition_file_content(model_file_content)
    return draw_net(
        caffe_net=model_params,
        rankdir='LR',
        ext='svg',
    )
