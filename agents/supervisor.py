from .react_agent import ReactAgent
from .agent_base import AgentConfig

class SupervisorAgent:
    def __init__(self, model_name, adapter):
        cfg = AgentConfig(model_name, adapter)
        self.react = ReactAgent(cfg)

    def run(self, prompt: str) -> str:
        response = self.react.run(prompt)
        # simple supervision: check length
        if len(response) < 20:
            response += "\n(Additional reasoning needed.)"
        return response