# Ref https://python.langchain.com/docs/how_to/custom_tools/
from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from typing import Annotated, List
from pydantic import BaseModel, Field
import asyncio
from langchain_core.tools import StructuredTool
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# toolの重要な属性
# name: 関数名
# description: docstringで書かれた関数の説明
# args_schema: Optionalだが設定するのが推奨される。callback handlerを使う場合は必須。
# return_direct: Agentのみに関係する。Trueなら、指定されたツールを呼び出した後、Agentは停止し結果をユーザに返す

def tool_with_decorator():
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

async def atool_with_decorator():
    @tool
    async def amultiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    print(await amultiply.ainvoke({"a": 2, "b": 3})) # 6

def tool_with_decorator_annotations():
    # annotationを解析することができる
    @tool
    def multiply_by_max(
        a: Annotated[int, "scale factor"],
        b: Annotated[List[int], "list of ints over which to take maximum"],
    ) -> int:
        """Multiply a by the maximum of b."""
        return a * max(b)
    
    print(multiply_by_max.invoke({"a": "2", "b": [3, 4]})) # 8
    print(multiply_by_max.args_schema.schema())
    # {'description': 'Multiply a by the maximum of b.', 
    # 'properties': {'a': {'description': 'scale factor',
    #   'title': 'A',
    #   'type': 'string'}, 
    # 'b': {'description': 'list of ints over which to take maximum', 
    #   'items': {'type': 'integer'}, 
    #   'title': 'B', 
    #   'type': 'array'}}, 
    # 'required': ['a', 'b'], 
    # 'title': 'multiply_by_max', 
    # 'type': 'object'}

def tool_with_decorator_some_args():
    class CalculatorInput(BaseModel):
        a: int = Field(description="first number")
        b: int = Field(description="second number")

    @tool("multiplication-tool", args_schema=CalculatorInput, return_direct=True)
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    # Let's inspect some of the attributes associated with the tool.
    print(multiply.name)        # multiplication-tool
    print(multiply.description) # Multiply two numbers.
    print(multiply.args)        # {'a': {'description': 'first number', 'title': 'A', 'type': 'integer'}, 'b': {'description': 'second number', 'title': 'B', 'type': 'integer'}}
    print(multiply.return_direct) # True
    print(multiply.invoke({"a": 2, "b": 3})) # 6

def tool_with_decorator_docstring():
    # docstringについては以下を参照
    # Ref https://google.github.io/styleguide/pyguide.html#383-functions-and-methods
    # 正しく設定しないとValueErrorになる
    @tool(parse_docstring=True)
    def foo(bar: str, baz: int) -> str:
        """The foo.

        Args:
            bar: The bar.
            baz: The baz.
        """
        return bar
    
    print(foo.args_schema.schema())
    # {'description': 'The foo.',
    # 'properties': {
    #   'bar': {'description': 'The bar.', 'title': 'Bar', 'type': 'string'}, 
    #   'baz': {'description': 'The baz.', 'title': 'Baz', 'type': 'integer'}
    # }, 
    # 'required': ['bar', 'baz'], 
    # 'title': 'foo', 
    # 'type': 'object'}


def tool_with_structuredTool():
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    async def amultiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    calculator = StructuredTool.from_function(func=multiply, coroutine=amultiply)
    
    print(calculator.invoke({"a": 2, "b": 3})) # 6
    print(asyncio.run(calculator.ainvoke({"a": 2, "b": 5}))) # 10

def tool_with_structuredTool_with_args_schema():
    class CalculatorInput(BaseModel):
        a: int = Field(description="first number")
        b: int = Field(description="second number")

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    async def amultiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        print(f"run amultiply!")
        return a * b

    calculator = StructuredTool.from_function(
        func=multiply,
        name="Calculator",
        description="multiply numbers",
        args_schema=CalculatorInput,
        return_direct=True,
        coroutine=amultiply # <- you can specify an async method if desired as well
    )

    print(calculator.invoke({"a": 2, "b": 3})) # 6
    print(calculator.name)        # Calculator
    print(calculator.description) # multiply numbers
    print(calculator.args) # {'a': {'description': 'first number', 'title': 'A', 'type': 'integer'}, 'b': {'description': 'second number', 'title': 'B', 'type': 'integer'}}
    print(asyncio.run(calculator.ainvoke({"a": 2, "b": 5}))) # run amultiply! 10

def create_dummy_tool_from_runnable():
    prompt = ChatPromptTemplate.from_messages(
        [("human", "Hello. Please respond in the style of {answer_style}.")]
    )

    # IFをテストするためのフェイクChatModel
    llm = GenericFakeChatModel(messages=iter(["hello matey"]))

    chain = prompt | llm | StrOutputParser()

    # as_tool()でツールに変換できる
    as_tool = chain.as_tool(
        name="Style responder", description="Description of when to use tool."
    )
    print(as_tool.args) # {'answer_style': {'title': 'Answer Style', 'type': 'string'}}
    print(as_tool.invoke({"answer_style": "foo"})) # hello matey

def create_yummy_tool_from_runnable():
    # 実際のChatModelでお試し
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
    print(snack_tool.args)          # {'country': {'title': 'Country', 'type': 'string'}, 'taste': {'title': 'Taste', 'type': 'string'}}
    print(snack_tool.return_direct) # False
    print(snack_tool.invoke({"taste": "sweet", "country": "USA"}))   # Funnel cake
    print(snack_tool.invoke({"taste": "salty", "country": "Japan"})) # Kaki no Tane

def create_custom_tool_with_BaseTool():
    from typing import Optional, Type

    from langchain_core.callbacks import (
        AsyncCallbackManagerForToolRun,
        CallbackManagerForToolRun,
    )
    from langchain_core.tools import BaseTool
    from pydantic import BaseModel, Field


    class CalculatorInput(BaseModel):
        a: int = Field(description="first number")
        b: int = Field(description="second number")

    # Note: It's important that every field has type hints. BaseTool is a
    # Pydantic class and not having type hints can lead to unexpected behavior.
    class CustomCalculatorTool(BaseTool):
        name: str = "Calculator"
        description: str = "useful for when you need to answer questions about math"
        args_schema: Type[BaseModel] = CalculatorInput
        return_direct: bool = True

        def _run(
            self,
            a: int,
            b: int,
            run_manager: Optional[CallbackManagerForToolRun] = None
        ) -> str:
            """Use the tool."""
            return a * b

        async def _arun(
            self,
            a: int,
            b: int,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        ) -> str:
            """Use the tool asynchronously."""
            # If the calculation is cheap, you can just delegate to the sync implementation
            # as shown below.
            # If the sync calculation is expensive, you should delete the entire _arun method.
            # LangChain will automatically provide a better implementation that will
            # kick off the task in a thread to make sure it doesn't block other async code.
            return self._run(a, b, run_manager=run_manager.get_sync())
    
    multiply = CustomCalculatorTool()
    print(multiply.name)         # Calculator
    print(multiply.description)  # useful for when you need to answer questions about math
    print(multiply.args)         # {'a': {'description': 'first number', 'title': 'A', 'type': 'integer'}, 'b': {'description': 'second number', 'title': 'B', 'type': 'integer'}}
    print(multiply.return_direct) # True

    print(multiply.invoke({"a": 2, "b": 3})) # 6
    print(asyncio.run(multiply.ainvoke({"a": 2, "b": 5}))) # 10

def create_async_tools_with_StructuredTool():
    from langchain_core.tools import StructuredTool

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    calculator = StructuredTool.from_function(func=multiply)

    print(calculator.invoke({"a": 2, "b": 3}))
    print(asyncio.run(calculator.ainvoke({"a": 2, "b": 5}))) # 別のスレッドに処理を委譲するため、小さなオーバーヘッドが発生する

def create_async_tools_with_StructuredTool_improved():
    from langchain_core.tools import StructuredTool

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    async def amultiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    calculator = StructuredTool.from_function(func=multiply, coroutine=amultiply)

    print(calculator.invoke({"a": 2, "b": 3}))
    print(asyncio.run(calculator.ainvoke({"a": 2, "b": 5}))) # amultiply を使うことで、スレッド処理によるオーバーヘッドを回避できる

def test_async_tool_sync_invoke_error():
    @tool
    async def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    # async の定義しかない tool で invoke を使うことはできないし、使うべきではない
    try:
        multiply.invoke({"a": 2, "b": 3})
    except NotImplementedError:
        print("Raised not implemented error. You should not be doing this.")

def error_handling_example():
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

    ### 文字列を指定する方法 ###
    get_weather_tool = StructuredTool.from_function(
        func=get_weather,
        handle_tool_error="There is no such city, but it's probably above 0K there!",
    )

    print(get_weather_tool.invoke({"city": "foobar"})) 
    # There is no such city, but it's probably above 0K there!

    def _handle_error(error: ToolException) -> str:
        return f"The following errors occurred during tool execution: `{error.args[0]}`"

    ### 関数を設定する方法 ###
    get_weather_tool = StructuredTool.from_function(
        func=get_weather,
        handle_tool_error=_handle_error,
    )

    print(get_weather_tool.invoke({"city": "foobar"})) 
    # The following errors occurred during tool execution: `Error: There is no city by the name of foobar.`

# ツールの出力をそのままモデルに渡すのではなく、出力の一部を渡したいとき
def return_artifacts_of_tool_execution():
    import random
    from typing import List, Tuple
    from langchain_core.tools import tool

    @tool(response_format="content_and_artifact")
    def generate_random_ints(min: int, max: int, size: int) -> Tuple[str, List[int]]:
        """Generate size random ints in the range [min, max]."""
        array = [random.randint(min, max) for _ in range(size)]
        content = f"Successfully generated array of {size} random ints in [{min}, {max}]."
        # (content, artifact) のタプルを返す
        return content, array
    
    # 単にツール引数でinvokeした場合は、contentだけが返される
    print(generate_random_ints.invoke({"min": 0, "max": 9, "size": 10}))
    # Successfully generated array of 10 random ints in [0, 9].

    # ToolCallとともに(?)invokeした場合は、contentとartifactの両方が入ったToolMessageが返される
    res = generate_random_ints.invoke(
        {
            "name": "generate_random_ints",
            "args": {"min": 0, "max": 9, "size": 10},
            "id": "123",          # required. 指定しないとKeyError
            "type": "tool_call",  # required. 指定しないとname属性が解釈できない(TypeError)
        }
    )

    print(type(res)) # <class 'langchain_core.messages.tool.ToolMessage'>
    print(res)
    # content='Successfully generated array of 10 random ints in [0, 9].' 
    # name='generate_random_ints' 
    # tool_call_id='123' 
    # artifact=[7, 7, 5, 0, 6, 3, 2, 4, 4, 1]
    print(f"res.content: {res.content}")   # res.content: Successfully generated array of 10 random ints in [0, 9].
    print(f"res.artifact: {res.artifact}") # res.artifact: [7, 7, 5, 0, 6, 3, 2, 4, 4, 1]

def return_artifacts_of_tool_execution_with_BaseTool():
    from langchain_core.tools import BaseTool
    from typing import List, Tuple
    import random

    class GenerateRandomFloats(BaseTool):
        name: str            = "generate_random_floats"
        description: str     = "Generate size random floats in the range [min, max]."
        response_format: str = "content_and_artifact"
        ndigits: int = 2

        def _run(self, min: float, max: float, size: int) -> Tuple[str, List[float]]:
            range_ = max - min
            array = [
                round(min + (range_ * random.random()), ndigits=self.ndigits)
                for _ in range(size)
            ]
            content = f"Generated {size} floats in [{min}, {max}], rounded to {self.ndigits} decimals."
            return content, array

        # Optionally define an equivalent async method

        # async def _arun(self, min: float, max: float, size: int) -> Tuple[str, List[float]]:
        #     ...
    
    rand_gen = GenerateRandomFloats(ndigits=4)

    res = rand_gen.invoke({"min": -1.0, "max": 1.0, "size": 2})
    print(type(res)) # str
    print(res)       # Generated 2 floats in [-1.0, 1.0], rounded to 4 decimals.

    toolMessage = rand_gen.invoke(
        {
            "name": "generate_random_floats",
            "args": {"min": 0.1, "max": 3.3333, "size": 3},
            "id": "123",
            "type": "tool_call",
        }
    )
    from langchain_core.messages.tool import ToolMessage
    print(type(toolMessage)) # <class 'langchain_core.messages.tool.ToolMessage'>
    print(f"toolMessage.content: {toolMessage.content}")   # res.content: Generated 3 floats in [0.1, 3.3333], rounded to 4 decimals.
    print(f"toolMessage.artifact: {toolMessage.artifact}") # res.artifact: [2.73, 1.0616, 1.6362]

def return_documents_as_artifacts():
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
    print(res.additional_kwargs) # {}
    print(res.response_metadata) # {}
            
if __name__ == "__main__":
    # @toolデコレータを使う方法
    tool_with_decorator()
    asyncio.run(atool_with_decorator())
    tool_with_decorator_annotations()
    tool_with_decorator_some_args()
    tool_with_decorator_docstring()

    # StructuredTool() を使う方法
    tool_with_structuredTool()
    tool_with_structuredTool_with_args_schema()

    # Runnable を tool にする方法
    create_dummy_tool_from_runnable()
    create_yummy_tool_from_runnable() # OpenAI API を使います

    # BaseTool から tool を作成
    create_custom_tool_with_BaseTool()

    # ===== async の tool を作るときの注意点 =====
    # 1. langchainはデフォルトで関数の計算コストが高いことを想定し、実行を別スレッドに委任する非同期実装を提供する
    # 2. async で実装を進めている場合は、オーバーヘッドを減らすために同期ツールの代わりに非同期ツールを使用すべき
    # 3. sync と async の両方の実装が必要な場合はStructuredToolかBaseToolを継承してツールを作る必要がある
    # 4. sync の実行がすぐに終わるなら、langchain の async をオーバーライドして単にsyncを実行する形にすればいい
    # 5. async の定義しかない tool で invoke を使うことはできないし、使うべきではない
    # ==========================================
    
    # toolの実行が軽いなら、langchainの提供する非同期実装ではなく、自前で定義した方がいい
    create_async_tools_with_StructuredTool()
    create_async_tools_with_StructuredTool_improved() # オーバーヘッドが少ない
    # test_async_tool_sync_invoke_error()               # async の定義しかない tool で invoke するとエラーになる例(注意点の5つ目)

    # handle_tool_errorの使い方
    error_handling_example()

    # ツールの出力をそのままモデルに渡すのではなく、出力の一部を渡したいとき
    # シンプルにinvokeしたときはcontentだけ、tool_callを指定した場合はcontentとartifactが受け取れる
    return_artifacts_of_tool_execution()
    return_artifacts_of_tool_execution_with_BaseTool()
    return_documents_as_artifacts() # Happy halloween!
