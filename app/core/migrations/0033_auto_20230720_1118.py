# Generated by Django 3.1 on 2023-07-20 11:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0032_auto_20230608_1016'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainedmodel',
            name='scaler_data',
            field=models.JSONField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='trainingmetrics',
            name='scaler_data',
            field=models.JSONField(blank=True, null=True),
        ),
    ]
