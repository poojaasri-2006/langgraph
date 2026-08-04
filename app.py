import os
import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from workflow import rt_app   # import compiled workflow

app = FastAPI(title="LangGraph Crew Workflow API")

# Expose workflow at /crew
add_routes(app, rt_app, path="/crew", playground_type="default")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
