import json
from dataclasses import dataclass
from typing import Any, Optional

from src.rag.retrieve import Retriever


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
    ):
        self.retriever = retriever if retriever is not None else Retriever(data_path)

        with open(test_queries_path, "r", encoding="utf-8") as f:
            self.test_payload = json.load(f)

        self.queries = self.test_payload["queries"]

    def evaluate(self, k: int = 3, verbose: bool = True) -> EvalMetrics:
        hit_sum = 0.0
        rr_sum = 0.0
        n = 0

        for q in self.queries:
            query_text = q["query"]
            gt_id = q["expected_recipe_id"]

            results = self.retriever.retrieve(query_text, number_of_docs=k)
            results = results or []  # None -> []

            hit = self.hit_at_k(results, gt_id)
            rr = self.rr_at_k(results, gt_id)

            hit_sum += hit
            rr_sum += rr
            n += 1

            if verbose:
                print(f"evaluating query: {query_text}")
                if results:
                    top_doc, top_score = results[0]
                    top_name = top_doc.metadata.get("recipe_name", "unknown").replace("Recipe Name : ", "").strip()
                    top_id = top_doc.metadata.get("recipe_id", "missing")
                    print(f"top: id={top_id} name={top_name} score={top_score}")
                else:
                    print("top: <no results>")
                print(f"hit@{k}={hit} rr@{k}={rr}")

                if hit == 0:
                    topk = []
                    for doc, score in results:
                        topk.append((doc.metadata.get("recipe_id", "missing"),
                                     doc.metadata.get("recipe_name", "unknown")
                                     .replace("Recipe Name : ", "").strip(),
                                     score))
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

    def hit_at_k(self, retrieval_results: list[tuple[Any, float]], gt_id: int) -> int:
        for doc, _score in retrieval_results:
            rid = self._safe_int(doc.metadata.get("recipe_id"))
            if rid is not None and rid == gt_id:
                return 1
        return 0

    def rr_at_k(self, retrieval_results: list[tuple[Any, float]], gt_id: int) -> float:
        for i, (doc, _score) in enumerate(retrieval_results, start=1):
            rid = self._safe_int(doc.metadata.get("recipe_id"))
            if rid is not None and rid == gt_id:
                return 1.0 / i
        return 0.0


def main():
    evaluator = RetrieverEvaluator()
    metrics = evaluator.evaluate(k=3, verbose=True)
    print("\nFINAL:", metrics)


if __name__ == "__main__":
    main()
