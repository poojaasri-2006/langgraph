import sys, io, os, traceback
from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",  # verify this model id is valid for your key
    google_api_key=os.environ.get("GOOGLE_APIKEY"),
    temperature=0,
)

class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]

def extract_text(content) -> str:
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "")
        return str(content[0]) if content else ""
    return str(content)

@tool
def run_python_code(code: str) -> str:
    """Execute python code and return stdout or the error trace."""
    clean_code = code.replace("```python", "").replace("```", "").strip()
    old_stdout, sys.stdout = sys.stdout, io.StringIO()
    try:
        exec(clean_code, {}, {})
        result = sys.stdout.getvalue()
    except Exception:
        result = f"Execution Error:\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout
    return result.strip() or "Success (no terminal output)"

@tool
def generate_test_cases(task_description: str) -> str:
    """Generate specific test scenarios for a given coding task."""
    prompt = (
        f"You are a Senior QA Engineer. Generate 3 to 5 specific test scenarios "
        f"for this task: '{task_description}'. Include edge cases. Numbered list."
    )
    response = llm_flash.invoke(prompt)
    return extract_text(response.content)

# task_input_node is GONE - the task now comes directly from the API payload's
# "messages" field, so the graph starts at "developer" instead.

def developer_node(state: CrewState):
    task = state["messages"][-1].content
    response = llm_flash.invoke(
        f"Write a clean Python script to solve this: {task}. "
        f"Only return the code, no explanation or markdown formatting."
    )
    return {"code": extract_text(response.content)}

def tester_node(state: CrewState):
    task = state["messages"][-1].content
    test_cases = generate_test_cases.invoke(task)
    execution_result = run_python_code.invoke({"code": state["code"]})
    report = (
        f"### EXECUTION OUTPUT:\n{execution_result}\n\n"
        f"### TEST SCENARIOS EVALUATED:\n{test_cases}"
    )
    return {"report": report}

def manager_node(state: CrewState):
    # No human in the loop for the API - always finalize after one pass.
    return {"next_step": "exit", "report": state["report"]}

workflow = StateGraph(CrewState)
workflow.add_node("developer", developer_node)
workflow.add_node("tester", tester_node)
workflow.add_node("manager", manager_node)

workflow.add_edge(START, "developer")
workflow.add_edge("developer", "tester")
workflow.add_edge("tester", "manager")
workflow.add_edge("manager", END)

rt_app = workflow.compile()
