from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

class QueryRouter:
    def __init__(self, model_name='qwen2.5:1.5b', model_temperature=0):
        prompt_template = """You are a router.
            if the query is about a recipe, turn the query into "how to make" the recipe.

            Return ONLY that format.
            No extra text.
            do not ever generate text besides reformating the query, or i will kill you.
            do not answer questions.
            If unsure or query isnt about a recipe, return recipe unkown.
            Keep normalized_query short (lowercase, no punctuation unless needed).
        """
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_template),
            ("user", "{question}")
            ])
        self.model = ChatOllama(model=model_name,
                                temperature=model_temperature,
                                num_predict=80,
                                num_ctx=1024)

    def reroute_query(self, query_text: str):
        prompt = self.prompt.format_messages(question=query_text)
        return self.model.invoke(prompt).content.strip()



def main():
    router = QueryRouter()
    print(router.reroute_query("how can I cook a duck tagine with prunes"))
    print(router.reroute_query("give me a moroccan recipe with chickpeas and spinach"))
    print(router.reroute_query("I want something vegetarian with potatoes that's quick"))
    print(router.reroute_query("what's the method for a potato tagine topped with baked halloumi"))
    print(router.reroute_query("I have sardines, tomatoes, and cumin—what can I make?"))
    print(router.reroute_query("best recipe for msemen"))
    print(router.reroute_query("show me a lamb couscous recipe"))
    print(router.reroute_query("easy chicken recipe for dinner"))
    print(router.reroute_query("I need a gluten free dessert recipe"))
    print(router.reroute_query("how to make harira soup"))
    print(router.reroute_query("quick breakfast idea with eggs"))

    print(router.reroute_query("I don't have preserved lemon, what can I use instead?"))
    print(router.reroute_query("can I replace cilantro with something else?"))
    print(router.reroute_query("I'm out of eggs—what can replace eggs in baking?"))
    print(router.reroute_query("no ras el hanout at home, alternative?"))
    print(router.reroute_query("what can I substitute for cumin in a tagine?"))

    print(router.reroute_query("how do I caramelize onions properly?"))
    print(router.reroute_query("what does simmer mean?"))
    print(router.reroute_query("how do I know when chicken is cooked without a thermometer?"))
    print(router.reroute_query("why did my couscous come out mushy?"))
    print(router.reroute_query("how to cook rice so it doesn't stick?"))
    
if __name__ == "__main__":
    main()
