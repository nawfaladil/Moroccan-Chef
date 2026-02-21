import os
import shutil
import json
from typing import List, Dict, Any

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "../../data")
chroma_path = os.path.join(data_dir, "chroma")

JSON_PATH = os.path.join(data_dir, "recipe_chunks.json")

def make_documents(json_path: str) -> List[Document]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        recipes: Dict[str, Any] = json.load(f)

    documents: List[Document] = []
    for recipe in recipes.values():
        rid = recipe.get("id")
        content = recipe.get("content", "")
        meta_list = recipe.get("metadatas", [])
        meta = meta_list[0] if (isinstance(meta_list, list) and len(meta_list) > 0) else {}

        # Normalize metadata
        meta = dict(meta) if isinstance(meta, dict) else {}
        meta["recipe_id"] = str(rid) if rid is not None else None

        documents.append(
            Document(
                page_content=content,
                metadata=meta
            )
        )

    return documents

def save_to_chroma(docs: List[Document], reset_db: bool = True) -> None:
    if reset_db and os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)

    model_name = "BAAI/bge-small-en-v1.5"
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}

    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    db = Chroma.from_documents(
        docs,
        embedding=hf,
        persist_directory=chroma_path
    )

    # Be explicit (safe across versions)
    try:
        db.persist()
    except Exception:
        pass

    print(f"Saved {len(docs)} recipes to {chroma_path} using device={device}.")

def generate_data_store():
    docs = make_documents(JSON_PATH)
    save_to_chroma(docs, reset_db=True)

def main():
    generate_data_store()

if __name__ == "__main__":
    main()
