# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('cnntools', '0010_caffecnnsnapshot_iteration'),
    ]

    operations = [
        migrations.RenameField(
            model_name='caffecnntrainingrun',
            old_name='deploy_file',
            new_name='deploy_file_snapshot',
        ),
    ]
