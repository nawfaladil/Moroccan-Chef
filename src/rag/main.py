from src.rag.generate import Generator
from src.rag.retrieve import Retriever

class Orchestrator:
    def __init__(self):
        self.generator = Generator()
        self.retriever = Retriever("C:/Users/nawfal/Documents/code/cooking_assistant/data")

    def respond(self, query_text: str):
        retrieval_results = self.retriever.retrieve(query_text)
        if retrieval_results:
            return self.generator.generate(query_text, retrieval_results)
        return None
