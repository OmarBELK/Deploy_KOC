# Generated by Django 3.1 on 2023-10-10 11:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0037_testingstatus_backtest_start_time'),
    ]

    operations = [
        migrations.AddField(
            model_name='testingstatus',
            name='progress',
            field=models.FloatField(default=0.0),
        ),
    ]
