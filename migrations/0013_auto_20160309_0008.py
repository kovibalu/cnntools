# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('cnntools', '0012_auto_20151008_2145'),
    ]

    operations = [
        migrations.AlterField(
            model_name='caffecnnsnapshot',
            name='training_run',
            field=models.ForeignKey(related_name='snapshots', on_delete=django.db.models.deletion.PROTECT, to='cnntools.CaffeCNNTrainingRun'),
            preserve_default=True,
        ),
        migrations.AlterField(
            model_name='caffecnntrainingrun',
            name='net',
            field=models.ForeignKey(related_name='training_runs', on_delete=django.db.models.deletion.PROTECT, to='cnntools.CaffeCNN'),
            preserve_default=True,
        ),
    ]
