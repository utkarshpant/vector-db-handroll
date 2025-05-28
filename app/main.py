from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="StackAI Take-Home")

@app.get('/health')
def health_check():
    return {'status': 'ok'}