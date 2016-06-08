# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import common.utils
import common.backends


class Migration(migrations.Migration):

    dependencies = [
        ('cnntools', '0011_auto_20151002_1430'),
    ]

    operations = [
        migrations.AlterField(
            model_name='caffecnnsnapshot',
            name='model_snapshot',
            field=models.FileField(storage=common.backends.DeconstructibleS3BotoStorage(), max_length=255, null=True, upload_to=common.utils.FileUploadPathKeepName(b'cnntools/snapshots', ext=b'caffemodel'), blank=True),
            preserve_default=True,
        ),
    ]
