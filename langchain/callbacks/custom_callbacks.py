# Ref https://python.langchain.com/docs/how_to/custom_callbacks/

# BaseCallbackHandlerがサポートするメソッドは以下を参照
# https://python.langchain.com/api_reference/core/callbacks/langchain_core.callbacks.base.BaseCallbackHandler.html#langchain-core-callbacks-base-basecallbackhandler

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts   import ChatPromptTemplate

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate

class MyCustomHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """ streaming=True のときのみ実行される """
        print(f"My custom handler, token: {token}")

def main():
    prompt = ChatPromptTemplate.from_messages(["Tell me a joke about {animal}"])

    # To enable streaming, we pass in `streaming=True` to the ChatModel constructor
    # Additionally, we pass in our custom handler as a list to the callbacks parameter
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        streaming=True, # 逐次的にトークンを取得するためにはTrueにする。Falseにするとon_llm_new_tokenは実行されない
        callbacks=[MyCustomHandler()],
    )

    chain = prompt | model

    response = chain.invoke({"animal": "bears"})
    print(response)

if __name__ == "__main__":
    main()