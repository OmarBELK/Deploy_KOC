#from core.models import Training
from core.models import MLmodel
from rest_framework import serializers


class MLmodelSerializer(serializers.ModelSerializer):
    file = serializers.FileField()
    class Meta:
        model = MLmodel
        fields = '__all__'


# class TrainingSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = Training
#         fields = ['servername','well_id','trained_at','training_status']