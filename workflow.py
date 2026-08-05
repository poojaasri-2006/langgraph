from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Initialize LLM with your API key from environment
llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=os.environ.get("GOOGLE_APIKEY"),
    temperature=0
)

class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]

def task_input_node(state: CrewState):
    return {"messages": [HumanMessage(content="Solve 2+2")], "next_step": "developer"}

def developer_node(state: CrewState):
    task = state['messages'][-1].content
    response = llm_flash.invoke(f"Write Python code to solve: {task}")
    return {"code": str(response.content)}

def tester_node(state: CrewState):
    code = state['code']
    return {"report": f"Code generated:\n{code}"}

def manager_node(state: CrewState):
    return {"next_step": "exit", "report": state["report"]}

workflow = StateGraph(CrewState)
workflow.add_node("task_input", task_input_node)
workflow.add_node("developer", developer_node)
workflow.add_node("tester", tester_node)
workflow.add_node("manager", manager_node)

workflow.add_edge(START, "task_input")
workflow.add_edge("task_input", "developer")
workflow.add_edge("developer", "tester")
workflow.add_edge("tester", "manager")
workflow.add_edge("manager", END)

rt_app = workflow.compile()
