import google.generativeai as genai
import os

API_KEY = os.environ.get("GOOGLE_API_KEY", "YOUR_API_KEY_HERE")
genai.configure(api_key=API_KEY)

print("Listing available models...")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
