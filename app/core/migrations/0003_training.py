# Generated by Django 3.2.3 on 2022-07-27 10:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0002_auto_20220517_1047'),
    ]

    operations = [
        migrations.CreateModel(
            name='Training',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('training_id', models.CharField(max_length=30)),
                ('well_id', models.CharField(max_length=10)),
                ('training_status', models.CharField(max_length=30)),
            ],
        ),
    ]
