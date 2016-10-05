# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('cnntools', '0002_caffecnntrainingrun_description'),
    ]

    operations = [
        migrations.AddField(
            model_name='caffecnn',
            name='html_short_description',
            field=models.TextField(default=b''),
            preserve_default=True,
        ),
    ]
