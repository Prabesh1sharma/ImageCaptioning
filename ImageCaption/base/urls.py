from django.urls import path
from . import views

urlpatterns = [
    path('',views.home),
    path('caption',views.captionbackend,name = 'caption'),
    

]

from django.conf import settings
from django.conf.urls.static import static
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)