from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Check if the environment variable is set
api_key = os.getenv("OPEN_AI_KEY")
print(f"API Key: {api_key}")

if api_key is None:
    raise ValueError("OPENAI_AI_KEY environment variable is not set.")
else:
    print("API Key loaded successfully.")
