# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('cnntools', '0009_auto_20151002_0344'),
    ]

    operations = [
        migrations.AddField(
            model_name='caffecnnsnapshot',
            name='iteration',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
    ]
