import json
from dataclasses import dataclass
from typing import Any, Optional
from langchain_core.documents import Document

from src.rag.retrieve import Retriever
from src.rag.route_query import QueryRouter


@dataclass
class EvalMetrics:
    hit_at_k: float
    mrr_at_k: float
    n: int


class RetrieverEvaluator:
    def __init__(
        self,
        test_queries_path: str = "C:/Users/nawfal/Documents/code/cooking_assistant/src/eval/evaluation_queries_50_paraphrased.json",
        data_path: str = "C:/Users/nawfal/Documents/code/cooking_assistant/data",
        retriever: Optional[Retriever] = None,
        k: int=3
    ):
        self.retriever = retriever if retriever is not None else Retriever(data_path, k=k)
        self.router = QueryRouter()

        with open(test_queries_path, "r", encoding="utf-8") as f:
            self.test_payload = json.load(f)

        self.queries = self.test_payload["queries"]
        self.k = k

    def evaluate(self, verbose: bool = True) -> EvalMetrics:
        hit_sum = 0.0
        rr_sum = 0.0
        n = 0

        for q in self.queries:
            query_text = q["query"]
            gt_id = q["expected_recipe_id"]
            print(f"evaluating query: {query_text}\n")
            query_text = self.router.reroute_query(query_text)
            print(f"rerouted query to : {query_text}\n")
            results = self.retriever.retrieve(query_text)
            results = results or []  # None -> []

            hit = self.hit_at_k(results, gt_id)
            rr = self.rr_at_k(results, gt_id)

            hit_sum += hit
            rr_sum += rr
            n += 1

            if verbose:
                top_doc = results[0]
                top_name = top_doc.metadata.get("recipe_name", "unknown").replace("Recipe Name : ", "").strip()
                top_id = top_doc.metadata.get("recipe_id", "missing")
                print(f"top: id={top_id} name={top_name}")
                print(f"hit@{self.k}={hit} rr@{self.k}={rr}")

                if hit == 0:
                    topk = []
                    for doc in results:
                        topk.append((doc.metadata.get("recipe_id", "missing"),
                                     doc.metadata.get("recipe_name", "unknown")
                                     .replace("Recipe Name : ", "").strip()))
                    print(f"MISS | gt_id={gt_id} | retrieved_ids={topk}")
                print("-----------")

        return EvalMetrics(
            hit_at_k=hit_sum / n if n else 0.0,
            mrr_at_k=rr_sum / n if n else 0.0,
            n=n,
        )

    @staticmethod
    def _safe_int(x: Any) -> Optional[int]:
        try:
            return int(x)
        except (TypeError, ValueError):
            return None

    def hit_at_k(self, retrieval_results: list[Document], gt_id: int) -> int:
        for doc in retrieval_results:
            rid = self._safe_int(doc.metadata.get("recipe_id"))
            if rid is not None and rid == gt_id:
                return 1
        return 0

    def rr_at_k(self, retrieval_results: list[Document], gt_id: int) -> float:
        for i, doc in enumerate(retrieval_results, start=1):
            rid = self._safe_int(doc.metadata.get("recipe_id"))
            if rid is not None and rid == gt_id:
                return 1.0 / i
        return 0.0


def main():
    evaluator = RetrieverEvaluator()
    metrics = evaluator.evaluate(verbose=True)
    print("\nFINAL:", metrics)


if __name__ == "__main__":
    main()
