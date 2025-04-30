from django.shortcuts import render
from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.views import APIView
from .utils.utils import genScoreResponse

# Create your views here.

class genScore(APIView):
	def get(self, request):
		print("GET GEN SCORE")
		return genScoreResponse(request)

	def post(self, request):
		print("POST GEN SCORE")
		return genScoreResponse(request)
	
class test(APIView):
	def get(self, request):
		print("GET TEST")
		return Response("Test GET response", status=status.HTTP_200_OK)
	
	def post(self, request):
		print("POST TEST")
		return Response("Test POST response", status=status.HTTP_200_OK)

class default(APIView):
	def get(self, request, resource):
		print("GET FORBIDDEN")
		return Response("Invalid request", status=status.HTTP_400_BAD_REQUEST)
	
	def post(self, request, resource):
		print("POST FORBIDDEN")
		return Response("Invalid request", status=status.HTTP_400_BAD_REQUEST)
