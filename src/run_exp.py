from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

print("KEY:", os.getenv("GROQ_API_KEY"))

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
print(client.models.list())
