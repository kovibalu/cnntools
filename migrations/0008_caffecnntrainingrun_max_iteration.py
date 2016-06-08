# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('cnntools', '0007_auto_20150722_1800'),
    ]

    operations = [
        migrations.AddField(
            model_name='caffecnntrainingrun',
            name='max_iteration',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
    ]
