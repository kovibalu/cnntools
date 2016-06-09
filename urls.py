from django.conf.urls import patterns, url

from .views import (caffe_cnn_detail, caffe_cnn_list,
                    caffe_cnn_snapshot_detail, caffe_cnn_trainingrun_dashboard,
                    caffe_cnn_trainingrun_detail)

urlpatterns = patterns(
    '',

    url(r'^$', caffe_cnn_list,
        name='cnntools-caffe-cnn-list'),
    url(r'^caffe_cnn_list/$', caffe_cnn_list,
        name='cnntools-caffe-cnn-list'),

    url(r'^caffe_cnn_detail/(?P<pk>\d+)/$', caffe_cnn_detail,
        name='cnntools-caffe-cnn-detail'),

    url(r'^caffe_cnn_trainingrun_detail/(?P<pk>\d+)/$', caffe_cnn_trainingrun_detail,
        name='cnntools-caffe-cnn-trainingrun-detail'),

    url(r'^caffe_cnn_snapshot_detail/(?P<pk>\d+)/$', caffe_cnn_snapshot_detail,
        name='cnntools-caffe-cnn-snapshot-detail'),

    url(r'^caffe_cnn_trainingrun_dashboard/$', caffe_cnn_trainingrun_dashboard,
        name='cnntools-caffe-cnn-trainingrun-dashboard'),
)

