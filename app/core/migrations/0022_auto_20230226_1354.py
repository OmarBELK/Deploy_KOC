# Generated by Django 3.1 on 2023-02-26 13:54

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0021_trainingstatus_model_name'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='trainingstatus',
            name='model_name',
        ),
        migrations.AddField(
            model_name='trainingmetrics',
            name='model_name',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
    ]
