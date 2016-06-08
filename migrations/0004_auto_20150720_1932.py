# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('cnntools', '0003_auto_20150719_2201'),
    ]

    operations = [
        migrations.AlterField(
            model_name='caffecnn',
            name='netid',
            field=models.CharField(unique=True, max_length=64),
            preserve_default=True,
        ),
    ]
