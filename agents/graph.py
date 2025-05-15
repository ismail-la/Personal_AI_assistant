from .supervisor import SupervisorAgent
from utils.config import MODEL_NAME, ADAPTER_NAME

# Initialize a single shared supervisor agent
ingent = SupervisorAgent(MODEL_NAME, ADAPTER_NAME)

def run_agent(prompt: str) -> str:
    """Runs the supervisor agent on a user prompt."""
    return ingent.run(prompt)