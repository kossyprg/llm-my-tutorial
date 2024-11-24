# Ref https://python.langchain.com/docs/how_to/output_parser_custom/
from dotenv import load_dotenv
load_dotenv()

from typing import Iterable
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.runnables import RunnableLambda


def main_use_runnable():
    model = ChatOpenAI(model="gpt-4o")

    def parse(ai_message: AIMessage) -> str:
        """Parse the AI message."""
        return ai_message.content.swapcase()

    # | を使用して parse をつなげた場合、自動的に Runnable(parse) に変換される。
    chain = model | parse
    res = chain.invoke("Just say Hello!", config={"run_name": "simple parse"})
    print(res) # hELLO!

    # 上とやっていることは同じ
    parse_runnable = RunnableLambda(parse)
    chain = model | parse_runnable
    res = chain.invoke("Just say Hello!", config={"run_name": "manually upgrade parse to Runnable"})
    print(res) # hELLO!

    # parser が出力を変換する前に、入力がまとまってしまうのでストリーミングできない。
    for chunk in chain.stream("tell me about yourself in one sentence"):
        print(chunk, end="|", flush=True)
    print()


    from langchain_core.runnables import RunnableGenerator

    def streaming_parse(chunks: Iterable[AIMessageChunk]) -> Iterable[str]:
        for chunk in chunks:
            yield chunk.content.swapcase()

    # ストリーミング機能を維持しながら、カスタム出力パーサーなどのカスタム動作を実装したいときは RunnableGenerator を使う
    streaming_parse = RunnableGenerator(streaming_parse)

    chain = model | streaming_parse
    for chunk in chain.stream("tell me about yourself in one sentence"):
        print(chunk, end="|", flush=True)
    print()
    # |i'M| AN| ai| LANGUAGE| MODEL| CREATED| BY| oPEN|ai|,| DESIGNED| TO| ASSIST| AND| PROVIDE| INFORMATION| ON| A| WIDE| RANGE| OF| TOPICS|.||

def main_inherit_from_parsing_base_classes_simple():
    from langchain_core.exceptions import OutputParserException
    from langchain_core.output_parsers import BaseOutputParser

    # The [bool] desribes a parameterization of a generic.
    # It's basically indicating what the return type of parse is.
    # In this case, the return type is either True or False
    class BooleanOutputParser(BaseOutputParser[bool]):
        """Custom boolean parser."""

        true_val: str = "YES"
        false_val: str = "NO"

        # モデルからの出力(文字列)を受け取って変換する
        def parse(self, text: str) -> bool:
            cleaned_text = text.strip().upper()
            if cleaned_text not in (self.true_val.upper(), self.false_val.upper()):
                raise OutputParserException(
                    f"BooleanOutputParser expected output value to either be "
                    f"{self.true_val} or {self.false_val} (case-insensitive). "
                    f"Received {cleaned_text}."
                )
            return cleaned_text == self.true_val.upper()

        # パーサーの名前
        @property
        def _type(self) -> str:
            return "boolean_output_parser"
    
    # 使用例
    parser = BooleanOutputParser()
    print(parser.invoke("YES")) # True

    # 不適な場合は例外が発生
    try:
        parser.invoke("MEOW")
    except Exception as e:
        print(f"Triggered an exception of type: {type(e)}")
    
    # パラメータを変更して実行
    parser = BooleanOutputParser(true_val="可", false_val="不可")
    print(parser.invoke("可"))

    # batch も使えることの確認
    print(parser.batch(["可", "不可", "不可"]))

    # モデルの出力をパース
    model = ChatOpenAI(model="gpt-4o")
    chain = model | parser
    print(chain.invoke("Say '可' or '不可'. You must NOT output any other words.")) # True

def main_inherit_from_parsing_base_classes_parse_raw_model_outputs():
    from typing import List

    from langchain_core.exceptions import OutputParserException
    from langchain_core.messages import AIMessage
    from langchain_core.output_parsers import BaseGenerationOutputParser
    from langchain_core.outputs import ChatGeneration, Generation

    class StrInvertCase(BaseGenerationOutputParser[str]):
        """An example parser that inverts the case of the characters in the message.

        This is an example parse shown just for demonstration purposes and to keep
        the example as simple as possible.
        """

        def parse_result(self, result: List[Generation], *, partial: bool = False) -> str:
            """Parse a list of model Generations into a specific format.

            Args:
                result: A list of Generations to be parsed. The Generations are assumed
                    to be different candidate outputs for a single model input.
                    Many parsers assume that only a single generation is passed it in.
                    We will assert for that
                partial: Whether to allow partial results. This is used for parsers
                        that support streaming
            """
            if len(result) != 1:
                raise NotImplementedError(
                    "This output parser can only be used with a single generation."
                )
            generation = result[0]
            if not isinstance(generation, ChatGeneration):
                # Say that this one only works with chat generations
                raise OutputParserException(
                    "This output parser can only be used with a chat generation."
                )
            return generation.message.content.swapcase()

    model = ChatOpenAI(model="gpt-4o")
    chain = model | StrInvertCase()
    print(chain.invoke("Tell me a short sentence about yourself"))
    # i'M AN ai DESIGNED TO ASSIST WITH A WIDE RANGE OF QUESTIONS AND TASKS, PROVIDING INFORMATION AND SUPPORT WHENEVER YOU NEED IT.

if __name__ == "__main__":
    main_use_runnable()

    # (非推奨) 大きなメリットがないのにコードの記述が増えるから
    main_inherit_from_parsing_base_classes_simple()
    main_inherit_from_parsing_base_classes_parse_raw_model_outputs()