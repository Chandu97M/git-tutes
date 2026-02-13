print("Hello Git Welcome")



import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Initialize environment and apps
load_dotenv()
app = FastAPI(title="GenAI GitHub Summarizer")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# --- 1. THE DATA SCHEMA ---
# Using Pydantic to ensure the AI output is 100% predictable
class IssueSummary(BaseModel):
    title: str = Field(description="The original issue title")
    bug_impact: str = Field(description="High-level impact of this bug on the system")
    severity: str = Field(pattern="^(Low|Medium|High)$")
    suggested_fix: str = Field(description="A technical code-level solution")

# --- 2. THE EXECUTION LOGIC ---
@app.get("/summarize/{owner}/{repo}/{issue_id}")
async def summarize_issue(owner: str, repo: str, issue_id: int):
    """
    Real-time Execution Flow:
    1. Fetch raw data from GitHub API
    2. Format text for the LLM
    3. Use 'JSON Mode' to get structured AI analysis
    4. Validate and return to the user
    """
    
    # Step A: Fetch from GitHub
    github_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_id}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    # Pro Tip: Using a token avoids 'Rate Limit' errors in production
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
        
    response = requests.get(github_url, headers=headers)
    
    if response.status_code != 200:
        raise HTTPException(status_code=404, detail="Could not find GitHub Issue")
        
    issue_data = response.json()
    content_to_analyze = f"Title: {issue_data['title']}\nDescription: {issue_data['body']}"

    # Step B: AI Processing (Agentic Logic)
    try:
        # We use 'gemini-2.0-flash' for the best balance of speed and cost
        ai_response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Summarize this bug report for a developer: {content_to_analyze}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=IssueSummary,
            ),
        )
        
        # Step C: Parse & Return
        # model_validate_json ensures the AI didn't 'hallucinate' the JSON format
        return IssueSummary.model_validate_json(ai_response.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")

# Health check for deployment platforms
@app.get("/health")
def health():
    return {"status": "online"}
