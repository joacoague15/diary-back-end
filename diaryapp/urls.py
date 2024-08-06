from django.urls import path
from . import views

urlpatterns = [
    path('rag/', views.rag_view, name='rag'),
    path('chat-completion/', views.chat_completion_view, name='chat_completion'),
    path('thread-completion/', views.thread_chat_completion_view, name='thread_completion'),
]