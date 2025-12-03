import google.generativeai as genai
import os

api_key = 'AIzaSyBH6QPeFgtZ1B26rt-p2GHaiD55IHEP6s0'
genai.configure(api_key=api_key)

print("Available models:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
