# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import common.utils
import common.backends


class Migration(migrations.Migration):

    dependencies = [
        ('cnntools', '0002_auto_20150719_1630'),
    ]

    operations = [
        migrations.AddField(
            model_name='caffecnntrainingrun',
            name='model_file_snapshot',
            field=models.FileField(storage=common.backends.DeconstructibleS3BotoStorage(), max_length=255, null=True, upload_to=common.utils.FileUploadPath(b'cnntools', ext=b'prototxt'), blank=True),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='caffecnntrainingrun',
            name='solver_file_snapshot',
            field=models.FileField(storage=common.backends.DeconstructibleS3BotoStorage(), max_length=255, null=True, upload_to=common.utils.FileUploadPath(b'cnntools', ext=b'prototxt'), blank=True),
            preserve_default=True,
        ),
    ]
