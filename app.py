import os
import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
from workflow import rt_app

app = FastAPI(title="LangGraph Crew API")

class AgentInput(BaseModel):
    input: str = Field(description="Enter your message")

def format_input(x):
    user_input = x["input"] if isinstance(x, dict) else x.input
    return {
        "messages": [
            {
                "type": "human",
                "content": user_input
            }
        ]
    }

chain = (
    RunnableLambda(format_input)
    | rt_app
)

add_routes(
    app,
    chain,
    path="/agent",
    playground_type="default"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
