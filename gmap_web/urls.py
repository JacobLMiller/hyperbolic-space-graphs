from django.conf.urls import include, url

import gmap_web.views as views

urlpatterns = [
	url(r'^admin$', views.home, name='home'),
	url(r'^admin/reload/$', views.rld, name='rld'),
	url(r'^', include(('maps.urls','display_map'), namespace="display_map")),
	#url(r'^accounts/login/$', 'django.contrib.auth.views.login'),
]
