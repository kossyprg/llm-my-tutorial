## Tools チュートリアル

Tools に関するチュートリアルを実行するためのソースファイル群です。

参考：[langchain Tools](https://python.langchain.com/docs/how_to/#tools)

## 実行方法

1. `.env` ファイルを作成して環境変数を記述してください。

```
OPENAI_API_KEY="<your-openai-api-key>"

# Langsmithでトレースする場合は以下4つが必要
# LANGCHAIN_PROJECTは任意の名前を設定できる
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="<your-langsmith-api-key>"
LANGCHAIN_PROJECT="tools-tutorial"
```

2. `Dockerfile` を使用してビルドします。

```bash
docker build -t tools .
```

3. ビルドしたイメージを実行してください。`-v`オプションでボリュームをマウントすると、ソースコードの修正がコンテナ環境にも反映されます。

Windows(cmd)の場合
```cmd
docker run -it --rm -v "%cd%":/home/user/app tools /bin/bash
```

4. 所望のスクリプトを実行してください。

```bash
python custom_tools.py
```

5. 終了する際は`exit`を入力してください

```bash
exit
```

## ソースコード

### Tools の作成方法
[custom_tools.py](custom_tools.py)

`tool`デコレータを使う方法がもっとも単純です。

```python
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# Let's inspect some of the attributes associated with the tool.
print(multiply.name)        # multiply
print(multiply.description) # Multiply two numbers.
print(multiply.args)        # {'a': {'title': 'A', 'type': 'integer'}, 'b': {'title': 'B', 'type': 'integer'}}
print(multiply.return_direct) # False
print(multiply.invoke({"a": 2, "b": 3})) # 6
```

`StructuredTool.from_function()`を制御できることが増えます。

```python
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

async def amultiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

calculator = StructuredTool.from_function(func=multiply, coroutine=amultiply)

print(calculator.invoke({"a": 2, "b": 3})) # 6
print(asyncio.run(calculator.ainvoke({"a": 2, "b": 5}))) # 10
```

文字列や`dict`を入力として受け付けるRunnableは`as_tool`メソッドでツールに変換できる。
```python
prompt = ChatPromptTemplate.from_messages(
    [("human", "Suggest a very {taste} snack in {country}. Just answer the name of the snack.")]
)

llm = ChatOpenAI(model="gpt-4o")
chain = prompt | llm | StrOutputParser()

# as_tool()でツールに変換できる
snack_tool = chain.as_tool(
    name="snack proposer", 
    description="Tool to suggest snacks in a specific country"
)

print(snack_tool.name)          # snack proposer
print(snack_tool.description)   # Tool to suggest snacks in a specific country
print(snack_tool.return_direct) # False
print(snack_tool.invoke({"taste": "sweet", "country": "USA"}))   # Funnel cake
print(snack_tool.invoke({"taste": "salty", "country": "Japan"})) # Kaki no Tane
```

`sync`の実装だけでも `ainvoke` を使えるが、いくつか注意点がある。
1. langchain はデフォルトで関数の計算コストが高いことを想定し、実行を別スレッドに委任する非同期実装を提供する
2. `async` で実装を進めている場合は、オーバーヘッドを減らすために同期ツールの代わりに非同期ツールを使用すべき
3. `sync` と `async` の両方の実装が必要な場合は `StructuredTool` か `BaseTool` を継承してツールを作る必要がある
4. `sync` の実行がすぐに終わるなら langchain の `async` をオーバーライドして単に `sync` を実行する形にすればいい
5. `async` の定義しかない `tool` で `invoke` しないでください（以下を参照）。

```python
@tool
async def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# async の定義しかない tool で invoke を使うことはできないし、使うべきではない
try:
    multiply.invoke({"a": 2, "b": 3}) # エラー
except NotImplementedError:
    print("Raised not implemented error. You should not be doing this.")
```

エラーは `ToolException` を投げるとよい

```python
from langchain_core.tools import ToolException

def get_weather(city: str) -> int:
    """Get weather for the given city."""
    raise ToolException(f"Error: There is no city by the name of {city}.")

### boolを設定する方法 ###
get_weather_tool = StructuredTool.from_function(
    func=get_weather,
    handle_tool_error=True,
)

print(get_weather_tool.invoke({"city": "foobar"})) 
# Error: There is no city by the name of foobar.
```

モデルに `Tool` の出力のすべてを渡す必要がないときには、`(content, artifact)` のタプルとして返すことができる。単に `invoke()` したときは `content` だけが返され、`ToolMessage` として受け取った時は `artifact` 属性に `artifact` が返される。

```python
from langchain_core.tools import BaseTool
from typing import Tuple
from langchain_core.documents import Document

class ReturnHalloweenDocument(BaseTool):
    name: str            = "Return_halloween_documents"
    description: str     = "Return the phrase for Halloween"
    response_format: str = "content_and_artifact"

    def _run(self, name: str) -> Tuple[str, str]:
        doc = Document(
            page_content="Trick or treat!", 
            metadata={"author": "kossy", "date": "2024-10-31"}
        )

        content = doc.metadata["date"]
        phrase = f"{name}! {doc.page_content}"
        return content, phrase

halloween = ReturnHalloweenDocument()

# metadata が返される
print(halloween.invoke({"name": "Ken"})) # 2024-10-31

# artifact として例の合言葉が返される
res = halloween.invoke(
    {
        "name": "Return_halloween_documents",
        "args": {"name": "Sakura"},
        "id": "20241031",
        "type": "tool_call"
    }
)

print(type(res)) # langchain_core.messages.tool.ToolMessage
print(res.content)  # 2024-10-31
print(res.artifact) # Sakura! Trick or treat!
print(res.tool_call_id) # 20241031
print(res.status)       # success
```

参考：
[How to create tools](https://python.langchain.com/docs/how_to/custom_tools/)
