from django.urls import path
from . import views

urlpatterns = [
    path('', views.recognize, name='img_rec')
]

