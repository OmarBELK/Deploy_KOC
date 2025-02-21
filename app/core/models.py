from django.db import models
from django.utils import timezone
from django.db import models
from django.utils.timezone import now
#from django.contrib.postgres.fields import JSONField
from django.db.models import JSONField

##from core.models import TrainingStatus

# Create your models here.

class TrainingStatus(models.Model):
    servername = models.CharField(max_length=100)
    wellid = models.CharField(max_length=100)
    status = models.CharField(max_length=100)
    created_at = models.DateTimeField(default=timezone.now)
    created_at_ms = models.BigIntegerField(null=True, blank=True)  # Store milliseconds timestamp
    user_id = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return f"{self.wellid} - {self.servername} - {self.status} - {self.created_at}"
    
class TrainingMetrics(models.Model):
    training_status = models.ForeignKey(TrainingStatus, on_delete=models.CASCADE, related_name='metrics')
    algo_name =  models.CharField(max_length=100, null=True, blank=True)
    pattern_ranges = JSONField(null=True, blank=True)  
    normal_ranges = JSONField(null=True, blank=True)
    model_name = models.CharField(max_length=100, null=True, blank=True)
    model_description = models.CharField(max_length=300, null=True, blank=True)
    is_generic = models.BooleanField(null=True, blank=True)
    inputs = models.TextField(null=True, blank=True)
    training_loss = models.FloatField(null=True, blank=True)
    training_accuracy = models.FloatField(null=True, blank=True)
    validation_loss = models.FloatField(null=True, blank=True)
    validation_accuracy = models.FloatField(null=True, blank=True)
    training_loss_values = models.TextField(null=True, blank=True)
    training_accuracy_values = models.TextField(null=True, blank=True)
    validation_loss_values = models.TextField(null=True, blank=True)
    validation_accuracy_values = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    created_at_ms = models.BigIntegerField(null=True, blank=True)  # Store milliseconds timestamp
    model_type = models.CharField(max_length=100, null=True, blank=True)
    creation_source = models.CharField(max_length=100, null=True, blank=True)
    last_trained = models.DateTimeField(null=True, blank=True)
    training_time = models.BigIntegerField(null=True, blank=True)
    execution_time = models.BigIntegerField(null=True, blank=True)



class TrainedModel(models.Model):
    algo_name =  models.CharField(max_length=100, null=True, blank=True)
    servername = models.CharField(max_length=100, null=True, blank=True)
    pattern_ranges = JSONField(null=True, blank=True)  
    normal_ranges = JSONField(null=True, blank=True)
    inputs = models.TextField(null=True, blank=True)
    wellid = models.CharField(max_length=100, null=True, blank=True)
    status = models.CharField(max_length=100 , null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    created_at_ms = models.BigIntegerField(null=True, blank=True)  # Store milliseconds timestamp
    model_name = models.CharField(max_length=100, null=True, blank=True)
    model_description = models.CharField(max_length=300, null=True, blank=True)
    training_loss = models.FloatField(null=True, blank=True)
    training_accuracy = models.FloatField(null=True, blank=True)
    validation_loss = models.FloatField(null=True, blank=True)
    validation_accuracy = models.FloatField(null=True, blank=True)
    training_loss_values = models.TextField(null=True, blank=True)
    training_accuracy_values = models.TextField(null=True, blank=True)
    validation_loss_values = models.TextField(null=True, blank=True)
    validation_accuracy_values = models.TextField(null=True, blank=True)
    trained_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    user_id = models.IntegerField(null=True, blank=True)
    model_type = models.CharField(max_length=100, null=True, blank=True)
    creation_source = models.CharField(max_length=100, null=True, blank=True)
    is_generic = models.BooleanField(null=True, blank=True)
    last_trained = models.DateTimeField(null=True, blank=True)
    training_time = models.BigIntegerField(null=True, blank=True)
    execution_time = models.BigIntegerField(null=True, blank=True)
    file_path = models.CharField(max_length=500, null=True, blank=True)

    def __str__(self):
        return f"{self.wellid} - {self.servername} - {self.status} - {self.created_at}"


class TestingStatus(models.Model):
    servername = models.CharField(max_length=100)
    wellid = models.CharField(max_length=100)
    status = models.CharField(max_length=100)
    created_at_ms = models.BigIntegerField(null=True, blank=True)
    trained_at_ms = models.BigIntegerField(null=True, blank=True)
    user_id = models.IntegerField(null=True, blank=True)
    backtest_start_time = models.DateTimeField(default=timezone.now)  
    progress = models.FloatField(default=0.0)  

    def __str__(self):
        return f"{self.wellid} - {self.servername} - {self.status} - {self.created_at_ms}"
