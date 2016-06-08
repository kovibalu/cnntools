# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('cnntools', '0006_auto_20150722_1207'),
    ]

    operations = [
        migrations.AlterField(
            model_name='caffecnntrainingrun',
            name='modified',
            field=models.DateTimeField(auto_now=True),
            preserve_default=True,
        ),
    ]
