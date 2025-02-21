from django.db import models

# Create your models here.
from django.db import models
from django.utils import timezone
from django.db import models
from django.utils.timezone import now
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
    model_name = models.CharField(max_length=100, null=True, blank=True)
    start = models.BigIntegerField(null=True, blank=True)
    end = models.BigIntegerField(null=True, blank=True)
    model_description = models.CharField(max_length=300, null=True, blank=True)
    is_generic = models.BooleanField(null=True, blank=True)
    inputs = models.TextField(null=True, blank=True)
    target_column = models.TextField(null=True, blank=True)
    performance = models.FloatField(null=True, blank=True)
    mse = models.FloatField(null=True, blank=True)
    y_scaler_value = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    created_at_ms = models.BigIntegerField(null=True, blank=True)  # Store milliseconds timestamp
    creation_source = models.CharField(max_length=100, null=True, blank=True)
    last_trained = models.DateTimeField(null=True, blank=True)
    training_time = models.BigIntegerField(null=True, blank=True)
    execution_time = models.BigIntegerField(null=True, blank=True)



class TrainedModel(models.Model):
    servername = models.CharField(max_length=100, null=True, blank=True) # not in metrics
    wellid = models.CharField(max_length=100, null=True, blank=True) # not in metrics
    start = models.BigIntegerField(null=True, blank=True)
    end = models.BigIntegerField(null=True, blank=True)
    status = models.CharField(max_length=100 , null=True, blank=True) # not in metrics
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    created_at_ms = models.BigIntegerField(null=True, blank=True)  # Store milliseconds timestamp
    model_name = models.CharField(max_length=100, null=True, blank=True)
    model_description = models.CharField(max_length=300, null=True, blank=True)
    inputs = models.TextField(null=True, blank=True)
    target_column = models.TextField(null=True, blank=True)
    performance = models.FloatField(null=True, blank=True)
    mse = models.FloatField(null=True, blank=True)
    y_scaler_value = models.FloatField(null=True, blank=True)
    trained_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    user_id = models.IntegerField(null=True, blank=True)
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
