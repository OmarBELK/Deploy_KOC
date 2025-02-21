from django.urls import path
from .views import feature_selection_correlation, feature_selection_importance
from .views import  start_training, check_training_status
from .views import start_backtest, check_backtesting_status,save_model, get_saved_models
from .views import predict
from .views import *


urlpatterns = [
    path('feature_selection_correlation/', feature_selection_correlation, name='feature_selection_correlation'),
    path('feature_selection_importance/', feature_selection_importance, name='feature_selection_importance'),
    path('start_training/', start_training, name='start_training_view'),
    path('check_training_status/', check_training_status, name='check_training_status'),
    path('start_backtesting/', start_backtest, name='start_backtesting_view'),
    path('check_backtesting_status/', check_backtesting_status, name='check_backtesting_status'),
    path('save_model/', save_model, name='save_model'),
    path('get_saved_models/', get_saved_models, name='get_saved_models'),
    path('get_saved_models_greater_than/', get_saved_models_greater_than, name='get_saved_models_greater_than'),
    path('delete_saved_model/', delete_saved_model, name='delete_saved_model'),
    path('predict/', predict, name='predict')
    
]
