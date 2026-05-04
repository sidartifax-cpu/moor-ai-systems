from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print("✅ Environment loaded successfully")
    print(f"🔑 API key found: {api_key[:8]}...")
else:
    print("❌ No API key found — check your .env file")
