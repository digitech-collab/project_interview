import requests
import markdown
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL_NAME = "llama-3.2-1b-instruct"
@csrf_exempt
def fetch_llm_response(request):
    """Django view to fetch LLM response and return JSON for AJAX."""
    
    # Handling GET and POST requests
    if request.method == "GET":
        Data = request.GET.get("Data", "")
    elif request.method == "POST":
        try:
            Data = json.loads(request.body).get("Data", "")
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=405)

    # Validate Data input
    if not Data:
        return JsonResponse({"error": "Data parameter is required"}, status=400)

    prompt = f"""
You are a **Posture & Speech Alignment Evaluator**, specializing in analyzing body language and speech delivery. 
Your task is to evaluate the following speaker's data and provide structured feedback.

### **Speaker Data:**
{Data}


"""

    # Payload for API request
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "stream": False
    }

    try:
        response = requests.post(LM_STUDIO_URL, json=data, timeout=10)  # Add timeout to prevent hanging requests

        if response.status_code == 200:
            result = response.json()
            message = result.get("choices", [{}])[0].get("message", {}).get("content", "No response received.")
            formatted_message = markdown.markdown(message)  # Convert Markdown to HTML
        else:
            formatted_message = f"<p style='color: red;'>Error: {response.status_code} - {response.text}</p>"

    except requests.exceptions.RequestException as e:
        formatted_message = f"<p style='color: red;'>Request failed: {e}</p>"

    return JsonResponse({"response_html": formatted_message})
