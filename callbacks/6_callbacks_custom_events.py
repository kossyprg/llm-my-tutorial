# Ref https://python.langchain.com/docs/how_to/callbacks_custom_events/

from dotenv import load_dotenv
load_dotenv()

from typing import Any, Dict, List, Optional
from uuid import UUID
import asyncio
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.callbacks.manager import (
    adispatch_custom_event,
)
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.config import RunnableConfig


async def example_adispatch_custom_event():
    # fooはRunnableになる
    @RunnableLambda
    async def foo(x: str) -> str:
        # adispatch_custom_eventを全てコメントアウトした場合、
        # astream_eventsで出力されるのはon_chain_start, on_chain_stream, on_chain_endのみ
        await adispatch_custom_event(name="event1", data={"x": x})
        await adispatch_custom_event(name="event2", data=5)
        await adispatch_custom_event(
            name="事件",
            data={"容疑者": "X", "被害者": "Y", "被害額": "$100", "発生日時": "2024-10-27-123456"}
        )
        return x
    
    # .stream()メソッドを使用した場合は最終出力のみが得られるのに対し、
    # .astream_events()メソッドは種々のイベントがイテレータとして返される
    # 参考: https://python.langchain.com/docs/concepts/streaming/#astream_events
    async for event in foo.astream_events("hello world", version="v2"):
        print(event)

"""
{'event': 'on_chain_start', 'data': {'input': 'hello world'}, 'name': 'foo', (omitted)}
{'event': 'on_custom_event', 'run_id': (omitted), 'name': 'event1', (omitted), 'data': {'x': 'hello world'}, (omitted)}
{'event': 'on_custom_event', 'run_id': (omitted), 'name': 'event2', (omitted), 'data': 5, (omitted)}
{'event': 'on_custom_event', 'run_id': (omitted), 'name': '事件', (omitted), 'data': {'容疑者': 'X', '被害者': 'Y', '被害額': '$100', '発生日時': '2024-10-27-123456'}, (omitted)}
{'event': 'on_chain_stream', 'run_id': (omitted), 'name': 'foo', (omitted), 'data': {'chunk': 'hello world'}, (omitted)}
{'event': 'on_chain_end', 'data': {'output': 'hello world'}, 'run_id': (omitted), 'name': 'foo', (omitted)}
"""

# Python 3.10以下の場合、RunnableConfigを明示的に渡してやる必要がある。
# Python 3.11以上なら必要ないが、互換性のために明示的に渡すのも悪くないとドキュメントに書かれている
# "If you are running python>=3.11, the RunnableConfig will automatically propagate to child runnables in async environment. However, it is still a good idea to propagate the RunnableConfig manually if your code may run in other Python versions."
async def example_adispatch_custom_event_under_python_3_10():
    @RunnableLambda
    async def bar(x: str, config: RunnableConfig) -> str:
        """An example that shows how to manually propagate config.

        You must do this if you're running python<=3.10.
        """
        await adispatch_custom_event("event1", {"x": x}, config=config)
        await adispatch_custom_event("event2", 5, config=config)
        await adispatch_custom_event("python version", {"version": "<=3.10"}, config=config)
        await adispatch_custom_event(
            name="事件", 
            data={"容疑者": "X", "被害者": "Y", "被害額": "$100", "発生日時": "2024-10-27-123456"}, 
            config=config
        )
        return x

    async for event in bar.astream_events("hello world", version="v2"):
        print(event)

async def via_async_callback_handler():
    class AsyncCustomCallbackHandler(AsyncCallbackHandler):
        async def on_custom_event(
            self,
            name: str,
            data: Any,
            *,
            run_id: UUID,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
        ) -> None:
            print('===========================')
            print(f"Received event: {name}")
            print(f"data: {data}")
            print(f"tags: {tags}")
            print(f"metadata: {metadata}")
            print(f"run_id: {run_id}")

    @RunnableLambda
    async def foo(x: str, config: RunnableConfig) -> str:
        """An example that shows how to manually propagate config.

        You must do this if you're running python<=3.10.
        """
        await adispatch_custom_event("event1", {"x": x}, config=config)
        await adispatch_custom_event("event2", 5, config=config)
        await adispatch_custom_event(
            name="事件", 
            data={"容疑者": "X", "被害者": "Y", "被害額": "$100", "発生日時": "2024-10-27-123456"}, 
            config=config
        )
        return x


    async_handler = AsyncCustomCallbackHandler()
    await foo.ainvoke(1, {"callbacks": [async_handler], "tags": ["foo", "bar"]})
"""
===========================
Received event: event1
data: {'x': 1}
tags: ['foo', 'bar']
metadata: {}
run_id: 188621d0-948f-45bc-b110-7d8676e8e7f1
===========================
Received event: event2
data: 5
tags: ['foo', 'bar']
metadata: {}
run_id: 188621d0-948f-45bc-b110-7d8676e8e7f1
===========================
Received event: 事件
data: {'容疑者': 'X', '被害者': 'Y', '被害額': '$100', '発生日時': '2024-10-27-123456'}
tags: ['foo', 'bar']
metadata: {}
run_id: 188621d0-948f-45bc-b110-7d8676e8e7f1
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.callbacks.manager import (
    dispatch_custom_event,
)
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.config import RunnableConfig


def via_sync_callback_handler():
    class CustomHandler(BaseCallbackHandler):
        def on_custom_event(
            self,
            name: str,
            data: Any,
            *,
            run_id: UUID,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
        ) -> None:
            print('===========================')
            print(f"Received event: {name}")
            print(f"data: {data}")
            print(f"tags: {tags}")
            print(f"metadata: {metadata}")
            print(f"run_id: {run_id}")

    @RunnableLambda
    def foo(x: int, config: RunnableConfig) -> int:
        dispatch_custom_event("event1", {"x": x})
        dispatch_custom_event("event2", {"x+3": x+3})
        dispatch_custom_event(
            name="事件", 
            data={"容疑者": "X", "被害者": "Y", "被害額": "$100", "発生日時": "2024-10-27-123456"}, 
            config=config
        )
        return x

    handler = CustomHandler()
    foo.invoke(2, {"callbacks": [handler], "tags": ["foo", "bar", "sync version"]})

"""
===========================
Received event: event1
data: {'x': 2}
tags: ['foo', 'bar', 'sync version']
metadata: {}
run_id: 0bacda78-8fed-4af1-9058-fc85f20e2904
===========================
Received event: event2
data: {'x+3': 5}
tags: ['foo', 'bar', 'sync version']
metadata: {}
run_id: 0bacda78-8fed-4af1-9058-fc85f20e2904
===========================
Received event: 事件
data: {'容疑者': 'X', '被害者': 'Y', '被害額': '$100', '発生日時': '2024-10-27-123456'}
tags: ['foo', 'bar', 'sync version']
metadata: {}
run_id: 0bacda78-8fed-4af1-9058-fc85f20e2904
"""

if __name__ == "__main__":
    # アタッチした自作のイベントを調べるコード
    asyncio.run(example_adispatch_custom_event())

    # # Python 3.10以下の場合
    # asyncio.run(example_adispatch_custom_event_under_python_3_10())

    # # async callback handlerを使って追加したイベントを確認する
    # asyncio.run(via_async_callback_handler())

    # # sync版
    # via_sync_callback_handler()