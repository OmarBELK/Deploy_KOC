# Generated by Django 3.1 on 2023-04-06 16:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0026_auto_20230406_0017'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainingmetrics',
            name='user_id',
            field=models.IntegerField(blank=True, null=True),
        ),
    ]
