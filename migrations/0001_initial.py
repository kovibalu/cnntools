# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import django.utils.timezone
import common.utils
import common.backends


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
                ('netid', models.CharField(max_length=16, db_index=True)),
                ('model_file', models.FileField(storage=common.backends.DeconstructibleS3BotoStorage(), max_length=255, null=True, upload_to=common.utils.FileUploadPath(b'cnntools', ext=b'prototxt'), blank=True)),
                ('solver_file', models.FileField(storage=common.backends.DeconstructibleS3BotoStorage(), max_length=255, null=True, upload_to=common.utils.FileUploadPath(b'cnntools', ext=b'prototxt'), blank=True)),
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
                ('outputs_json', models.TextField()),
                ('output_names_json', models.TextField()),
                ('net', models.ForeignKey(related_name='training_runs', to='cnntools.CaffeCNN')),
            ],
            options={
                'ordering': ['-id'],
                'abstract': False,
            },
            bases=(models.Model,),
        ),
    ]
