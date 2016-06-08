# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('cnntools', '0005_caffecnntrainingrun_modified'),
    ]

    operations = [
        migrations.AlterField(
            model_name='caffecnntrainingrun',
            name='modified',
            field=models.DateTimeField(default=django.utils.timezone.now, auto_now=True),
            preserve_default=True,
        ),
    ]
