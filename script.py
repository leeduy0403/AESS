# import pprint
# import google.generativeai as genai
# from load_creds import load_creds

# creds = load_creds()

# genai.configure(credentials=creds)

# print(creds)
# print('Available base models:', [m.name for m in genai.list_models()])
import json


with open("input.json", 'r', encoding='utf-8') as f:
    raw_data = f.read()
    print("Raw JSON Data:", raw_data)  # Debugging line
    data = json.loads(raw_data)
