import os
import json
import requests
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# --- INITIALIZATION ---
load_dotenv()
app = FastAPI(title="Production GitHub Summarizer")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Connect to Redis (Ensure your Redis server is running!)
redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True)

# Define the structured output
class IssueSummary(BaseModel):
    title: str
    bug_summary: str = Field(description="One sentence high-level summary.")
    severity: str = Field(description="Low, Medium, or High.")
    suggested_fix: str = Field(description="Detailed technical solution.")

# --- THE RETRY LOGIC (THE SECRET SAUCE) ---
@retry(
    stop=stop_after_attempt(3), # Try 3 times total
    wait=wait_exponential(multiplier=1, min=4, max=10), # Wait 4s, then 8s, then 10s
    retry=retry_if_exception_type(Exception), # In production, filter for specific API errors
    reraise=True # If all 3 fail, show the original error
)
async def call_gemini_with_retry(prompt_text: str):
    """Encapsulated AI call with automatic recovery."""
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt_text,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=IssueSummary,
        ),
    )
    return response.text

# --- THE MAIN ENDPOINT ---
@app.get("/summarize/{owner}/{repo}/{issue_id}")
async def summarize_issue(owner: str, repo: str, issue_id: int):
    # 1. Check Cache
    cache_key = f"summary:{owner}:{repo}:{issue_id}"
    cached = await redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # 2. Fetch from GitHub
    github_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_id}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token: headers["Authorization"] = f"token {token}"
    
    resp = requests.get(github_url, headers=headers)
    if resp.status_code != 200:
        raise HTTPException(status_code=404, detail="GitHub Issue not found")
    
    issue_data = resp.json()
    prompt = f"Analyze this GitHub issue: {issue_data['title']} \n {issue_data['body']}"

    # 3. Call AI with Retry logic
    try:
        raw_json = await call_gemini_with_retry(prompt)
        
        # 4. Save to Cache & Return
        await redis_client.set(cache_key, raw_json, ex=86400) # Save for 24 hours
        return json.loads(raw_json)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI failure after 3 retries: {str(e)}")
