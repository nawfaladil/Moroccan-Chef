from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

class Generator:
    def __init__(self, model_name='qwen3:4b', model_temperature=0):
        self.model = ChatOllama(model=model_name, temperature=model_temperature)
        prompt_template = """
        You are a cooking assistant.

        Use ONLY the context below.
        select the single most relevant recipe.

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
        self.prompt_template = ChatPromptTemplate.from_template(prompt_template)

    def generate(self, query_text: str, retrieval_results: list[Document]):
        context_text = "\n\n---\n\n".join(
            f"{doc.page_content}" for doc in retrieval_results
        )
        prompt = self.prompt_template.format(context=context_text, question=query_text)
        print(prompt)
        sources = [doc.metadata.get("recipe_name", "unknown").replace("Recipe Name : ", "")
                   .strip() for doc in retrieval_results]
        model_response = self.model.invoke(prompt)
        return (model_response.content, sources)
