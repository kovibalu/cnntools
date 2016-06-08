# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('cnntools', '0004_auto_20150720_1932'),
    ]

    operations = [
        migrations.AddField(
            model_name='caffecnntrainingrun',
            name='modified',
            field=models.DateTimeField(default=django.utils.timezone.now),
            preserve_default=True,
        ),
    ]
