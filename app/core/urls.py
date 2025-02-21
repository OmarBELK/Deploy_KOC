from django.urls import path
from core.views import server_capacity
from core.views import save_model
from core.views import check_training_status
from core.views import start_training
from core.views import delete_saved_model
from core.views import get_saved_models
from core.views import view_synthetic_data
from core.views import get_saved_models_greater_than
from core.views import predict
from core.views import start_backtest, check_backtest_status
from core.views import predict_manual
from core.views import run_manual_autoencoder

from core.views import start_backtest_admin, check_backtest_status_admin

urlpatterns = [
    path("view_synthetic_data/", view_synthetic_data),
    path("start_training/", start_training),
    path("check_training_status/", check_training_status),
    path("start_backtest/",start_backtest), # corrected 
    path("check_backtest_status/",check_backtest_status),
    path("save_model/", save_model),
    path("get_saved_models/",get_saved_models),
    path("delete_saved_model/",delete_saved_model),
    path("server_capacity/", server_capacity),
    path("get_saved_models_greater_than/", get_saved_models_greater_than),
    path("predict/", predict), # corrected
    path("predict_manual/",predict_manual),
    path("run_manual_autoencoder/", run_manual_autoencoder), # corrected

    path("start_backtest_admin/", start_backtest_admin),
    path("check_backtest_status_admin/", check_backtest_status_admin)
    
]
