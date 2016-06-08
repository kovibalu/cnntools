# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import common.backends
import django.utils.timezone
import django.db.models.deletion
import cnntools.common_utils


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='CaffeCNN',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('added', models.DateTimeField(default=django.utils.timezone.now)),
                ('description', models.TextField()),
                ('netid', models.CharField(unique=True, max_length=64)),
                ('model_file', models.FileField(storage=common.backends.DeconstructibleS3BotoStorage(), max_length=255, null=True, upload_to=cnntools.common_utils.FileUploadPath(b'cnntools', ext=b'prototxt'), blank=True)),
                ('solver_file', models.FileField(storage=common.backends.DeconstructibleS3BotoStorage(), max_length=255, null=True, upload_to=cnntools.common_utils.FileUploadPath(b'cnntools', ext=b'prototxt'), blank=True)),
                ('deploy_file', models.FileField(storage=common.backends.DeconstructibleS3BotoStorage(), max_length=255, null=True, upload_to=cnntools.common_utils.FileUploadPath(b'cnntools', ext=b'prototxt'), blank=True)),
            ],
            options={
                'ordering': ['-id'],
                'abstract': False,
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='CaffeCNNSnapshot',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('added', models.DateTimeField(default=django.utils.timezone.now)),
                ('iteration', models.IntegerField()),
                ('sha1', models.CharField(max_length=40)),
                ('model_snapshot', models.FileField(storage=common.backends.DeconstructibleS3BotoStorage(), max_length=255, null=True, upload_to=cnntools.common_utils.FileUploadPathKeepName(b'cnntools/snapshots', ext=b'caffemodel'), blank=True)),
            ],
            options={
                'ordering': ['-id'],
                'abstract': False,
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='CaffeCNNTrainingRun',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('added', models.DateTimeField(default=django.utils.timezone.now)),
                ('final_iteration', models.IntegerField()),
                ('max_iteration', models.IntegerField()),
                ('model_file_snapshot', models.FileField(storage=common.backends.DeconstructibleS3BotoStorage(), max_length=255, null=True, upload_to=cnntools.common_utils.FileUploadPath(b'cnntools', ext=b'prototxt'), blank=True)),
                ('solver_file_snapshot', models.FileField(storage=common.backends.DeconstructibleS3BotoStorage(), max_length=255, null=True, upload_to=cnntools.common_utils.FileUploadPath(b'cnntools', ext=b'prototxt'), blank=True)),
                ('deploy_file_snapshot', models.FileField(storage=common.backends.DeconstructibleS3BotoStorage(), max_length=255, null=True, upload_to=cnntools.common_utils.FileUploadPath(b'cnntools', ext=b'prototxt'), blank=True)),
                ('outputs_json', models.TextField()),
                ('output_names_json', models.TextField()),
                ('modified', models.DateTimeField(auto_now=True)),
                ('net', models.ForeignKey(related_name='training_runs', on_delete=django.db.models.deletion.PROTECT, to='cnntools.CaffeCNN')),
            ],
            options={
                'ordering': ['-id'],
                'abstract': False,
            },
            bases=(models.Model,),
        ),
        migrations.AddField(
            model_name='caffecnnsnapshot',
            name='training_run',
            field=models.ForeignKey(related_name='snapshots', on_delete=django.db.models.deletion.PROTECT, to='cnntools.CaffeCNNTrainingRun'),
            preserve_default=True,
        ),
    ]
