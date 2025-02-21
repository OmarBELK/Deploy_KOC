# Generated by Django 3.1 on 2023-03-31 02:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0024_auto_20230226_1400'),
    ]

    operations = [
        migrations.DeleteModel(
            name='EvaluationMetrics',
        ),
        migrations.DeleteModel(
            name='MLmodel',
        ),
        migrations.DeleteModel(
            name='TrainingResult',
        ),
        migrations.RemoveField(
            model_name='trainedmodel',
            name='description',
        ),
        migrations.RemoveField(
            model_name='trainedmodel',
            name='model_data',
        ),
        migrations.AddField(
            model_name='trainedmodel',
            name='created_at',
            field=models.DateTimeField(auto_now_add=True, null=True),
        ),
        migrations.AddField(
            model_name='trainedmodel',
            name='model_description',
            field=models.CharField(blank=True, max_length=300, null=True),
        ),
        migrations.AddField(
            model_name='trainedmodel',
            name='status',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AddField(
            model_name='trainedmodel',
            name='trained_at',
            field=models.DateTimeField(auto_now_add=True, null=True),
        ),
        migrations.AddField(
            model_name='trainedmodel',
            name='training_accuracy',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='trainedmodel',
            name='training_accuracy_values',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='trainedmodel',
            name='training_loss',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='trainedmodel',
            name='training_loss_values',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='trainedmodel',
            name='validation_accuracy',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='trainedmodel',
            name='validation_accuracy_values',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='trainedmodel',
            name='validation_loss',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='trainedmodel',
            name='validation_loss_values',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='trainedmodel',
            name='model_name',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='trainedmodel',
            name='servername',
            field=models.CharField(max_length=100),
        ),
        migrations.AlterField(
            model_name='trainedmodel',
            name='wellid',
            field=models.CharField(max_length=100),
        ),
    ]
