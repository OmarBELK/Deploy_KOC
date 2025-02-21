# Generated by Django 3.1 on 2023-02-26 13:42

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0019_trainingstatus_status'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='trainingstatus',
            name='completed_at',
        ),
        migrations.RemoveField(
            model_name='trainingstatus',
            name='started_at',
        ),
        migrations.AddField(
            model_name='trainingstatus',
            name='created_at',
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
        migrations.AlterField(
            model_name='trainingstatus',
            name='status',
            field=models.CharField(max_length=100),
        ),
        migrations.AlterField(
            model_name='trainingstatus',
            name='wellid',
            field=models.CharField(max_length=100),
        ),
    ]
