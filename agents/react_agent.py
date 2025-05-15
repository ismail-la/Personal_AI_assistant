# react_agent.py

from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline

class ReactAgent:
    def __init__(self, config):
        self.generator = pipeline("text-generation", model=config.model_name)

    def run(self, prompt: str) -> str:
        result = self.generator(prompt, max_length=150, do_sample=True)[0]["generated_text"]
        return result
