## Callbackチュートリアル

langchainに書かれているCallbacksに関するチュートリアルを実行するためのソースファイル群です。公式ドキュメントでは`ChatAnthropic`を使用していますが、このチュートリアルでは`ChatOpenAI`を使用します。

参考：[langchain Callbacks](https://python.langchain.com/docs/how_to/#callbacks)

## 実行方法

1. `.env` ファイルを作成して `OPENAI_API_KEY` を記述してください。

```
OPENAI_API_KEY="YOUR_API_KEY"
```

2. `Dockerfile` を使用してビルドします。

```bash
docker build -t callbacks .
```

3. ビルドしたイメージを実行してください。`-v`オプションでボリュームをマウントすると、ソースコードの修正がコンテナ環境にも反映されます。

Windows(cmd)の場合
```cmd
docker run -it --rm -v "%cd%":/home/user/app callbacks /bin/bash
```

4. 所望のスクリプトを実行してください。

```bash
python callbacks_runtime.py
```

5. 終了する際は`exit`を入力してください

```bash
exit
```

## ソースコード

### 1. 実行時にcallbacksを渡す方法
[callbacks_runtime.py](callbacks_runtime.py)

`invoke` 時にcallbacksを渡す方法です。`Agent`に渡すと、`Agent`自体のみでなく、関連するツールやLLMにもコールバックが使用されます。

```python
chain = prompt | llm 
chain.invoke({"number": "2"}, config={"callbacks": callbacks})
```

参考：
[How to pass callbacks in at runtime](https://python.langchain.com/docs/how_to/callbacks_runtime/)

### 2. callbackをRunnableにアタッチする方法
[callbacks_attach.py](callbacks_attach.py)

`.with_config()`メソッドを使ってRunnableにアタッチする方法です。`invoke`時に毎回callbackを渡す手間がなくなります。

```python
chain = prompt | llm 
chain_with_callbacks = chain.with_config(callbacks=callbacks)
chain_with_callbacks.invoke({"number": "2"})
```

参考：
[How to attach callbacks to a runnable](https://python.langchain.com/docs/how_to/callbacks_attach/)

### 3. コンストラクタに渡す方法
[callbacks_constructor.py](callbacks_constructor.py)

```python
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    callbacks=callbacks, # ここで渡す
)

chain = prompt | llm
chain.invoke({"number": "2"})
```

このスクリプトの例では、`on_chat_model_start` と `on_llm_end` のコールバックのみが実行され、`on_chain_start` と `on_chain_end` が実行されないことに注意が必要です。

出力例
```
Chat model started
Chat model ended, response: generations=[(omitted)] llm_output={(omitted)} run=None type='LLMResult'
```

参考：
[How to propagate callbacks constructor](https://python.langchain.com/docs/how_to/callbacks_constructor/)

### 4. callbacksを自作する
[custom_callbacks.py](custom_callbacks.py)

```python
class MyCustomHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """ streaming=True のときのみ実行される """
        print(f"My custom handler, token: {token}")
```

自作したコールバックを渡す
```python
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    streaming=True, # on_llm_new_tokenを実行するためにはTrue必須
    callbacks=[MyCustomHandler()],
)
```

BaseCallbackHandlerがサポートするメソッドは以下を参照。

[BaseCallbackHandler](https://python.langchain.com/api_reference/core/callbacks/langchain_core.callbacks.base.BaseCallbackHandler.html#langchain-core-callbacks-base-basecallbackhandler)


参考：[How to create custom callback handlers](https://python.langchain.com/docs/how_to/custom_callbacks/)

### 5. 非同期処理にコールバックを使う方法
[callbacks_async.py](callbacks_async.py)

イベントがブロックされるのを防ぐためには`AsyncCallbackHandler`を使うのが良いです。
```python
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
```

自作したコールバックを渡す
```python
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
await asyncio.gather(llm_task, countdown_task)
```

出力例（カウントダウン処理とLLMのイベントが並列処理されています）
```
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
```

参考：[How to use callbacks in async environments](https://python.langchain.com/docs/how_to/callbacks_async/)

### 6. 自作のイベントをディスパッチする方法
[callbacks_custom_events.py](callbacks_custom_events.py)

```python
async def example_adispatch_custom_event():
    # fooはRunnableになる
    @RunnableLambda
    async def foo(x: str) -> str:
        await adispatch_custom_event(name="event1", data={"x": x})
        await adispatch_custom_event(name="event2", data=5)
        await adispatch_custom_event(
            name="事件",
            data={"容疑者": "X", "被害者": "Y", "被害額": "$100", "発生日時": "2024-10-27-123456"}
        )
        return x
    
    # .stream()メソッドを使用した場合は最終出力のみが得られるのに対し、
    # .astream_events()メソッドは種々のイベントがイテレータとして返される
    async for event in foo.astream_events("hello world", version="v2"):
        print(event)
```

出力
```
{'event': 'on_chain_start', 'data': {'input': 'hello world'}, 'name': 'foo', (omitted)}
{'event': 'on_custom_event', 'run_id': (omitted), 'name': 'event1', (omitted), 'data': {'x': 'hello world'}, (omitted)}
{'event': 'on_custom_event', 'run_id': (omitted), 'name': 'event2', (omitted), 'data': 5, (omitted)}
{'event': 'on_custom_event', 'run_id': (omitted), 'name': '事件', (omitted), 'data': {'容疑者': 'X', '被害者': 'Y', '被害額': '$100', '発生日時': '2024-10-27-123456'}, (omitted)}
{'event': 'on_chain_stream', 'run_id': (omitted), 'name': 'foo', (omitted), 'data': {'chunk': 'hello world'}, (omitted)}
{'event': 'on_chain_end', 'data': {'output': 'hello world'}, 'run_id': (omitted), 'name': 'foo', (omitted)}
```

参考：[How to dispatch custom callback events](https://python.langchain.com/docs/how_to/callbacks_custom_events/)