from rest_framework.response import Response
from rest_framework import status
import json

def generateResult(data):
	descriptions_urls = []
	rubric_urls = []
	submission_urls = []
	results = []

	# Get the submission urls, need download later
	if "descriptions" in data:
		for description in data["descriptions"]:
			descriptions_urls.append(description["description_urls"])

	# Get the rubric urls, need download later
	if "rubrics" in data:
		for rubric in data["rubrics"]:
			rubric_urls.append(rubric["rubric_urls"])
		
	if "submissions" in data:
		for submission in data["submissions"]:
			submission_urls.append(submission["submission_urls"])
			result = {}
			result["submission_id"] = submission["submission_id"]

			# Get the submission urls, need download
			submission_files = submission["submission_urls"]
			#* Replace with actual scoring logic
			result["scores"] = [8, 9, 10]
			result["feedbacks"] = ["Great", "Great", "Great"]
			result["components"] = ["Part 1", "Part 2", "Part 3"]

			results.append(result)

	return results

def genScoreResponse(request):
	data = json.loads(request.body)
	try:
		result = generateResult(data=data)
	except Exception as e:
		return Response(json.dumps({"error": str(e)}), status=status.HTTP_500_INTERNAL_SERVER_ERROR)

	response = {}
	response["results"] = result

	return Response(json.dumps(response), status=status.HTTP_200_OK)
