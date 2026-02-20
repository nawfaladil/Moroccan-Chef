import os
import argparse
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
# from langchain_huggingface import HuggingFaceBgeEmbeddings
from langchain_ollama import ChatOllama


script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '../../data')

CHROMA_PATH = data_dir+"/chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    # model_name = "BAAI/bge-small-en-v1.5"
    # model_kwargs = {"device": "cpu"}
    # encode_kwargs = {"normalize_embeddings": True}
    # hf = HuggingFaceBgeEmbeddings(
    #     model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    # )
    # embedding_function = hf
    db = Chroma(persist_directory=CHROMA_PATH)

    # Search the DB.
    results = db.similarity_search(query_text, k=3)
    # if len(results) == 0 or results[0][1] < 0.7:
    #     print(f"Unable to find matching results.")
    #     return

    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOllama(model='qwen3:4b')
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()