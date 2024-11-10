from dotenv import load_dotenv
load_dotenv()

from langfuse.callback import CallbackHandler
from langchain_core.runnables import RunnableLambda
import time
import os

def callback_example():
    # callback handler の設定
    # Ref https://langfuse.com/docs/get-started#log-your-first-llm-call-to-langfuse
    langfuse_handler = CallbackHandler(
        secret_key=os.environ['LANGFUSE_SECRET_KEY'],
        public_key=os.environ['LANGFUSE_PUBLIC_KEY'],
        host=os.environ['LANGFUSE_HOST'],
    )

    # Runnable.stream() については以下を参照
    # Ref https://python.langchain.com/docs/how_to/lcel_cheatsheet/#stream-a-runnable
    def yield_characters(s: str):
        for char in s:
            yield char # return だと stream() したときに最初の1文字で終わってしまう
    
    runnable = RunnableLambda(yield_characters)

    # (c) Undertale
    for chunk in runnable.stream(
        input="ケツイがみなぎった。",
        config={"callbacks": [langfuse_handler], "run_name": "ニンゲン"}
    ):
        print(chunk, end="", flush=True) # flush=True は出力をバッファにためないための措置
        time.sleep(0.1)

    print() # 改行したいだけ

if __name__ == "__main__":
    # callbackを設定する方法
    # Ref https://langfuse.com/docs/get-started#log-your-first-llm-call-to-langfuse
    callback_example()
    
