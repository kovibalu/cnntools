# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('cnntools', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='caffecnn',
            name='netid',
            field=models.CharField(unique=True, max_length=16),
            preserve_default=True,
        ),
    ]
