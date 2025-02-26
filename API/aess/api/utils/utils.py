from rest_framework.response import Response
from rest_framework import status
import json
from .eval import evaluate_submissions

def genScoreResponse(request):
	data = json.loads(request.body)
	try:
		response = evaluate_submissions(data=data)
		return Response(json.dumps(response), status=status.HTTP_200_OK)
	except Exception as e:
		return Response(json.dumps({"error": str(e)}), status=status.HTTP_500_INTERNAL_SERVER_ERROR)
