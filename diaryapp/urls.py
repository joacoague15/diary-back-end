from django.urls import path
from . import views

urlpatterns = [
    path('rag/', views.rag_view, name='rag'),
    path('chat-completion/', views.chat_completion_view, name='chat_completion'),
    path('character-system-role/', views.character_system_role_view, name='character_system_role'),
    path('character-responses/', views.chat_responses_view, name='character_responses'),
]