# Generated by Django 3.1 on 2024-08-14 11:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0043_auto_20240715_1925'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainedmodel',
            name='normal_ranges',
            field=models.JSONField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='trainingmetrics',
            name='normal_ranges',
            field=models.JSONField(blank=True, null=True),
        ),
    ]
