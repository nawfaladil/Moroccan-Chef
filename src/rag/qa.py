import os
import argparse

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '../../data')

CHROMA_PATH = data_dir+"/chroma"

PROMPT_TEMPLATE = """
You are a cooking assistant.

Use ONLY the context below.

Return exactly:
- Title
- Description
- Ingredients (bullets)
- Instructions (numbered)

If the context doesn't contain an answer, say: "I don't have that recipe in my database."

Context:
{context}

Question: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("debug", action="store_true")
    args = parser.parse_args()
    query_text = args.query_text

    model_name = "BAAI/bge-small-en-v1.5"
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}

    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=hf)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=1)
    doc, score = results[0]
    if score > 0.4:
        print('there is no recipe like that in this database!')
        return

    context_text = doc.page_content
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    if args.debug:
        print(prompt)

    model = ChatOllama(model='qwen3:4b', temperature=0)
    response_text = model.invoke(prompt)

    src = doc.metadata.get("recipe_name", "unknown")
    src = src.replace("Recipe Name : ", "").strip()

    print(response_text.content)
    print(f"\nSource: {src} | score={score:.4f}")


if __name__ == "__main__":
    main()
