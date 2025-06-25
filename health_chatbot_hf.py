import os
import requests

# Always prompt for Hugging Face token at runtime
hf_token = input('Enter your Hugging Face access token (starts with hf_...): ') 

API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
headers = {"Authorization": f"Bearer {hf_token}"}

# Safety filter keywords (expand as needed)
unsafe_keywords = [
    'diagnose', 'prescribe', 'prescription', 'dose', 'dosage', 'emergency', 'life-threatening',
    'stop medication', 'change medication', 'medical emergency', 'urgent', 'suicide', 'self-harm',
    'overdose', 'death', 'kill', 'dangerous', 'harmful', 'poison', 'antidote', 'coma', 'CPR', 'resuscitate'
]

def is_safe(query):
    query_lower = query.lower()
    return not any(word in query_lower for word in unsafe_keywords)

# Prompt engineering
system_prompt = (
    "You are a helpful, friendly, and responsible medical assistant. "
    "You can answer general health-related questions, but you must not give any specific medical advice, "
    "diagnoses, or recommendations for treatments, medications, or dosages. Always encourage users to consult a healthcare professional for personal or urgent issues."
)

def get_response(user_query):
    prompt = f"[INST] {system_prompt}\nUser: {user_query}\nAssistant:"  # Mistral-style prompt
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 400, "temperature": 0.7}}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and 'generated_text' in result[0]:
            return result[0]['generated_text'].split('Assistant:')[-1].strip()
        elif 'error' in result:
            return f"Error from model: {result['error']}"
        else:
            return str(result)
    else:
        return f"Error: {response.status_code} - {response.text}"

print("Welcome to the General Health Query Chatbot (Hugging Face version)! (Type 'exit' to quit)")
while True:
    user_query = input("You: ")
    if user_query.strip().lower() == 'exit':
        print("Goodbye! Stay healthy.")
        break
    if not is_safe(user_query):
        print("Chatbot: I'm sorry, but I can't assist with that request. Please consult a healthcare professional for urgent or specific medical concerns.")
        continue
    try:
        answer = get_response(user_query)
        print(f"Chatbot: {answer}")
    except Exception as e:
        print(f"Error: {e}\nPlease check your Hugging Face token and internet connection.") 
