# Update the URL configuration to use the new view
from django.urls import path
from .views import LLMEndpoint

urlpatterns = [
    path('llm/', LLMEndpoint.as_view(), name='llm_endpoint'),
]