import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import importlib.util
import sys

# Define API models
class RunRequest(BaseModel):
    inputs: dict = {}

class RunResponse(BaseModel):
    result: dict

# Initialize FastAPI app
app = FastAPI(title="CrewAI Deployment Server")

# Load crew and flow modules
def load_module(module_path, module_name):
    if not os.path.exists(module_path):
        raise ImportError(f"Module file {module_path} not found")
        
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load configuration
with open("deployment_config.json", "r") as f:
    config = json.load(f)

# Initialize crews and flows
crews = {}
flows = {}

for crew_config in config.get("crews", []):
    name = crew_config["name"]
    module_path = crew_config["module_path"]
    class_name = crew_config["class_name"]
    
    module = load_module(module_path, f"crew_{name}")
    crew_class = getattr(module, class_name)
    crews[name] = crew_class()
    
for flow_config in config.get("flows", []):
    name = flow_config["name"]
    module_path = flow_config["module_path"]
    class_name = flow_config["class_name"]
    
    module = load_module(module_path, f"flow_{name}")
    flow_class = getattr(module, class_name)
    flows[name] = flow_class()

# Define API endpoints
@app.get("/")
def read_root():
    return {"status": "running", "crews": list(crews.keys()), "flows": list(flows.keys())}

@app.post("/run/crew/{crew_name}")
def run_crew(crew_name: str, request: RunRequest):
    if crew_name not in crews:
        raise HTTPException(status_code=404, detail=f"Crew '{crew_name}' not found")
        
    try:
        crew_instance = crews[crew_name].crew()
        result = crew_instance.kickoff(inputs=request.inputs)
        return {"result": {"raw": result.raw}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run/flow/{flow_name}")
def run_flow(flow_name: str, request: RunRequest):
    if flow_name not in flows:
        raise HTTPException(status_code=404, detail=f"Flow '{flow_name}' not found")
        
    try:
        flow_instance = flows[flow_name]
        result = flow_instance.kickoff(inputs=request.inputs)
        return {"result": {"value": str(result)}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
