from django.urls import path
from . import views

urlpatterns = [
	path("gen-score/", views.genScore.as_view(), name="eval"),
	path("test/", views.test.as_view(), name="test"),
	path("", views.default.as_view(), name="default"),  # Default path
	path("<path:resource>", views.default.as_view(), name="wildcard"),  # Wildcard path
]