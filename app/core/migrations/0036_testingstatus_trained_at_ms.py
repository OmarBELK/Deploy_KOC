# Generated by Django 3.1 on 2023-08-25 11:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0035_testingstatus'),
    ]

    operations = [
        migrations.AddField(
            model_name='testingstatus',
            name='trained_at_ms',
            field=models.BigIntegerField(blank=True, null=True),
        ),
    ]
