from typing import TypedDict, List, Optional
import os

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ.get("GOOGLE_APIKEY"),
    temperature=0
)


class CrewState(TypedDict):
    messages: List[BaseMessage]
    code: Optional[str]
    report: Optional[str]


def developer_node(state: CrewState):
    user_message = state["messages"][-1].content

    response = llm.invoke(
        f"Write Python code for the following task:\n\n{user_message}"
    )

    return {
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "code": response.content,
    }


def tester_node(state: CrewState):
    report = f"Generated Code:\n\n{state['code']}"

    return {
        "messages": state["messages"] + [AIMessage(content=report)],
        "report": report,
    }


workflow = StateGraph(CrewState)

workflow.add_node("developer", developer_node)
workflow.add_node("tester", tester_node)

workflow.add_edge(START, "developer")
workflow.add_edge("developer", "tester")
workflow.add_edge("tester", END)

rt_app = workflow.compile()
