from openai import OpenAI
from os import getenv
from dotenv import load_dotenv

load_dotenv()

__all__ = ["client"]

client = OpenAI(
    api_key=getenv("OPENAI_API_KEY")
)