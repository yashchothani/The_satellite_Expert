import os
from dotenv import load_dotenv


def load_environment():
    load_dotenv()
    return os.getenv("GROQ_API_KEY")