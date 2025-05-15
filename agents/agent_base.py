from abc import ABC, abstractmethod

class AgentConfig:
    def __init__(self, model_name: str, peft_adapter: str):
        self.model_name = model_name
        self.peft_adapter = peft_adapter

class AgentBase(ABC):
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg

    @abstractmethod
    def run(self, prompt: str) -> str:
        """Generate a response to the prompt."""
        pass