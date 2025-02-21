from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/',admin.site.urls),
    path('automl/',include('core.urls')),
    path('autopr/',include('corepredict.urls'))
]
