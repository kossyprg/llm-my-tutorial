# Ref https://python.langchain.com/docs/how_to/callbacks_constructor/

from dotenv import load_dotenv
load_dotenv()

from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages  import BaseMessage
from langchain_core.outputs   import LLMResult
from langchain_core.prompts   import ChatPromptTemplate

class LoggingHandler(BaseCallbackHandler):
    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs
    ) -> None:
        print("Chat model started")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        print(f"Chat model ended, response: {response}")

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs
    ) -> None:
        if serialized is None:
            name = "None"
        else:
            name = serialized.get("name")
        print(f"Chain {name} started")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        print(f"Chain ended, outputs: {outputs}")

def main():
    # オブジェクトの子には継承されず、混乱を招くので、通常はinvoke時の引数に渡す方がよいとされる。
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        callbacks=[LoggingHandler()], # ここで渡す
    )

    prompt = ChatPromptTemplate.from_template("What is 1 + {number}?")

    chain = prompt | llm

    # `on_chat_model_start` と `on_llm_end` のコールバックのみが実行され、`on_chain_start` と `on_chain_end` が実行されないことに注意。
    chain.invoke({"number": "2"})


if __name__ == "__main__":
    main()