# Ref https://python.langchain.com/docs/how_to/callbacks_async/

from dotenv import load_dotenv
load_dotenv()

from typing import Any, Dict, List
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult


class MyCustomSyncHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"Sync handler being called in a `thread_pool_executor`: token: {token}")

class MyCustomAsyncHandler(AsyncCallbackHandler):
    """Async callback handler that can be used to handle callbacks from langchain."""

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""
        print("zzzz....")
        await asyncio.sleep(3) # 3 sec
        class_name = serialized["name"]
        print(f"Hi! I just woke up. Your llm is starting: {class_name}")

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when chain ends running."""
        print("zzzz....")
        await asyncio.sleep(3)
        print("Hi! I just woke up. Your llm is ending")

# 非同期処理を明示的に実感できるようにするためのカウントダウン処理
# awaitによって処理の合間に別のタスクが動くようになる
async def perform_countdown(count: int):
    """A simple countdown function to demonstrate parallel execution."""
    while count > 0:
        print(f"Countdown: {count}")
        await asyncio.sleep(1)
        count -= 1
    print("Countdown finished!")

async def main():
    # To enable streaming, we pass in `streaming=True` to the ChatModel constructor
    # Additionally, we pass in a list with our custom handler
    chat = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        streaming=True, # 逐次的にトークンを取得するためにはTrueにする
        callbacks=[MyCustomSyncHandler(), MyCustomAsyncHandler()],
    )

    llm_task = chat.agenerate([[HumanMessage(content="Tell me a joke")]])
    countdown_task = perform_countdown(5)

    # LLMの出力生成とカウントダウンが同時に行われる
    await asyncio.gather(llm_task, countdown_task)

"""
Countdown: 5
zzzz....
Countdown: 4
Countdown: 3
Hi! I just woke up. Your llm is starting: ChatOpenAI
Countdown: 2
Sync handler being called in a `thread_pool_executor`: token:
Sync handler being called in a `thread_pool_executor`: token: Why
Sync handler being called in a `thread_pool_executor`: token:  don't
Sync handler being called in a `thread_pool_executor`: token:  skeleton
Sync handler being called in a `thread_pool_executor`: token: s
Sync handler being called in a `thread_pool_executor`: token:  fight
Sync handler being called in a `thread_pool_executor`: token:  each
Sync handler being called in a `thread_pool_executor`: token:  other
Sync handler being called in a `thread_pool_executor`: token: ?


Sync handler being called in a `thread_pool_executor`: token: They
Sync handler being called in a `thread_pool_executor`: token:  don't
Sync handler being called in a `thread_pool_executor`: token:  have
Sync handler being called in a `thread_pool_executor`: token:  the
Sync handler being called in a `thread_pool_executor`: token:  guts
Sync handler being called in a `thread_pool_executor`: token: !
Sync handler being called in a `thread_pool_executor`: token:
zzzz....
Countdown: 1
Countdown finished!
Hi! I just woke up. Your llm is ending
"""

if __name__ == "__main__":
    asyncio.run(main())