from src.rag.generate import Generator
from src.rag.retrieve import Retriever
from src.rag.route_query import QueryRouter

class Orchestrator:
    def __init__(self, k: int = 1):
        self.generator = Generator()
        self.router = QueryRouter()
        self.retriever = Retriever(k=k)

    def respond(self, query_text: str):
        routed_query = self.router.reroute_query(query_text)
        retrieval_results = self.retriever.retrieve(routed_query)
        if retrieval_results:
            return self.generator.generate(query_text, retrieval_results)
        return None
