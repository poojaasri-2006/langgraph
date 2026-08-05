import os
import uvicorn

from fastapi import FastAPI
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from workflow import rt_app

app = FastAPI(title="LangGraph Crew API")


class AgentInput(BaseModel):
    input: str = Field(
        description="Enter your prompt"
    )


def format_input(x):
    if isinstance(x, dict):
        user_input = x["input"]
    else:
        user_input = x.input

    return {
        "messages": [
            HumanMessage(content=user_input)
        ]
    }


def extract_output(state):
    messages = state.get("messages", [])

    if messages:
        return messages[-1].content

    return "No response generated."


chain = (
    RunnableLambda(format_input)
    | rt_app
    | RunnableLambda(extract_output)
).with_types(
    input_type=AgentInput,
    output_type=str
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
