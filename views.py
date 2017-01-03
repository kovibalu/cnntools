import json

from cnntools import utils
from cnntools.models import CaffeCNN, CaffeCNNSnapshot, CaffeCNNTrainingRun
from cnntools.snapshot_utils import load_net_from_snapshot
from django.contrib.admin.views.decorators import staff_member_required
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, render


@staff_member_required
def caffe_cnn_list(
        request,
        template='cnntools/caffe_cnn_list.html'):

    entries = CaffeCNN.objects.all()
    context = {
        'nav': 'cnntools',
        'subnav': 'models',
        'entries': entries,
    }

    return render(request, template, context)


@staff_member_required
def caffe_cnn_detail(
        request,
        pk,
        template='cnntools/caffe_cnn_detail.html'):

    caffe_cnn = get_object_or_404(CaffeCNN, pk=pk)
    model_file_content = caffe_cnn.get_model_file_content()
    solver_file_content = caffe_cnn.get_solver_file_content()
    deploy_file_content = caffe_cnn.get_deploy_file_content()

    # sections on the page
    nav_section_keys = [
        ("description", 'Description'),
        ("trainruns", 'Training Runs'),
        ("model_file", 'Model File Content'),
        ("solver_file", 'Solver File Content'),
        ("deploy_file", 'Deploy File Content'),
    ]
    nav_sections = [
        {
            'key': t[0],
            'name': t[1],
            'template': 'cnntools/caffecnn/%s.html' % t[0],
        }
        for t in nav_section_keys if t
    ]
    context = {
        'nav': 'cnntools',
        'subnav': 'models',
        'nav_sections': nav_sections,
        'caffe_cnn': caffe_cnn,
        'model_file_content': model_file_content,
        'solver_file_content': solver_file_content,
        'deploy_file_content': deploy_file_content,
    }

    return render(request, template, context)


@staff_member_required
def caffe_cnn_trainingrun_detail(
        request,
        pk,
        template='cnntools/caffe_cnn_trainingrun_detail.html'):

    caffe_cnn_trrun = get_object_or_404(CaffeCNNTrainingRun, pk=pk)
    outputs, output_names = caffe_cnn_trrun.get_outputs()
    #svgs = utils.get_svgs_from_output(outputs, output_names)
    figures = utils.get_figures_from_output(outputs, output_names)

    model_file_content = caffe_cnn_trrun.get_model_file_content()
    solver_file_content = caffe_cnn_trrun.get_solver_file_content()
    deploy_file_content = caffe_cnn_trrun.get_deploy_file_content()

    # sections on the page
    nav_section_keys = [
        ("description", 'Description'),
        ("figures", 'Figures'),
        ("model_file", 'Model File Content'),
        ("deploy_file", 'Deploy File Content'),
        ("solver_file", 'Solver File Content'),
        ("snapshots", 'Snapshots'),
    ]
    nav_sections = [
        {
            'key': t[0],
            'name': t[1],
            'template': 'cnntools/caffecnn_trrun/%s.html' % t[0],
        }
        for t in nav_section_keys if t
    ]
    context = {
        'nav': 'cnntools',
        'subnav': 'models',
        'nav_sections': nav_sections,
        'caffe_cnn_trrun': caffe_cnn_trrun,
        'model_file_content': model_file_content,
        'solver_file_content': solver_file_content,
        'deploy_file_content': deploy_file_content,
        'figures': figures,
        'figures_json': json.dumps(figures),
    }

    return render(request, template, context)


@staff_member_required
def caffe_cnn_trainingrun_dashboard(
        request,
        template='cnntools/caffe_cnn_trainingrun_dashboard.html'):

    # TODO: Paginate this page, because we might want to see more training runs...
    recent_trruns = CaffeCNNTrainingRun.objects.\
        order_by('-modified').all()[:5]

    all_svgs = []
    for trrun in recent_trruns:
        outputs, output_names = trrun.get_outputs()
        svgs = utils.get_svgs_from_output(outputs, output_names, simplified=True)
        all_svgs.append(svgs)

    entries = zip(recent_trruns, all_svgs)

    context = {
        'nav': 'cnntools',
        'subnav': 'trainingrun-dashboard',
        'entries': entries,
    }

    return render(request, template, context)


@staff_member_required
def caffe_cnn_snapshot_detail(
        request,
        pk,
        template='cnntools/caffe_cnn_snapshot_detail.html'):

    snapshot = get_object_or_404(CaffeCNNSnapshot, pk=pk)
    net = load_net_from_snapshot(snapshot.id)
    weight_plots = utils.get_svgs_from_net(net)

    # sections on the page
    nav_section_keys = [
        ("description", 'Description'),
        ("figures", 'Figures'),
    ]
    nav_sections = [
        {
            'key': t[0],
            'name': t[1],
            'template': 'cnntools/caffecnn_snapshot/%s.html' % t[0],
        }
        for t in nav_section_keys if t
    ]
    context = {
        'nav': 'cnntools',
        'subnav': 'models',
        'nav_sections': nav_sections,
        'snapshot': snapshot,
        'caffe_cnn_trrun': snapshot.training_run,
        'weight_plots': weight_plots,
    }

    return render(request, template, context)


@staff_member_required
def caffe_cnn_graph(request, pk):
    entry = get_object_or_404(CaffeCNN, pk=pk)
    return HttpResponse(entry.get_net_graph_svg(), content_type='image/svg+xml')


@staff_member_required
def caffe_cnn_trainingrun_graph(request, pk):
    entry = get_object_or_404(CaffeCNNTrainingRun, pk=pk)
    return HttpResponse(entry.get_net_graph_svg(), content_type='image/svg+xml')

