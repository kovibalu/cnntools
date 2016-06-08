# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import django.utils.timezone
import common.utils
import common.backends


class Migration(migrations.Migration):

    dependencies = [
        ('cnntools', '0008_caffecnntrainingrun_max_iteration'),
    ]

    operations = [
        migrations.CreateModel(
            name='CaffeCNNSnapshot',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('added', models.DateTimeField(default=django.utils.timezone.now)),
                ('sha1', models.CharField(max_length=40)),
                ('model_snapshot', models.FileField(storage=common.backends.DeconstructibleS3BotoStorage(), max_length=255, null=True, upload_to=common.utils.FileUploadPath(b'cnntools/snapshots', ext=b'caffemodel'), blank=True)),
                ('training_run', models.ForeignKey(related_name='snapshots', to='cnntools.CaffeCNNTrainingRun')),
            ],
            options={
                'ordering': ['-id'],
                'abstract': False,
            },
            bases=(models.Model,),
        ),
        migrations.AddField(
            model_name='caffecnn',
            name='deploy_file',
            field=models.FileField(storage=common.backends.DeconstructibleS3BotoStorage(), max_length=255, null=True, upload_to=common.utils.FileUploadPath(b'cnntools', ext=b'prototxt'), blank=True),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='caffecnntrainingrun',
            name='deploy_file',
            field=models.FileField(storage=common.backends.DeconstructibleS3BotoStorage(), max_length=255, null=True, upload_to=common.utils.FileUploadPath(b'cnntools', ext=b'prototxt'), blank=True),
            preserve_default=True,
        ),
    ]
