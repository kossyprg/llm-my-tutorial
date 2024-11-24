# Ref https://python.langchain.com/docs/how_to/output_parser_structured/
from dotenv import load_dotenv
load_dotenv()

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from pydantic import BaseModel, Field, model_validator
from typing import List, ClassVar

def main_pydantic_output_parser():
    model = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.0)

    # Define your desired data structure.
    class SightSeeingSpotsInJapan(BaseModel):
        prefecture: str = Field(description="旅行先の県名")
        spots: List[str] = Field(description="その県の有名な観光地")

        # インスタンス変数でないことを示すときに ClassVar を使う
        # Ref https://docs.python.org/ja/3/library/typing.html#typing.ClassVar
        JAPAN_PREFERCTURES: ClassVar[List[str]] = [
            "北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県",
            "茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県",
            "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県", "岐阜県",
            "静岡県", "愛知県", "三重県", "滋賀県", "京都府", "大阪府", "兵庫県",
            "奈良県", "和歌山県", "鳥取県", "島根県", "岡山県", "広島県", "山口県",
            "徳島県", "香川県", "愛媛県", "高知県", "福岡県", "佐賀県", "長崎県",
            "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県"
        ]

        # You can add custom validation logic easily with Pydantic.
        @model_validator(mode="before")
        @classmethod
        def check_prefecture(cls, values: dict) -> dict:
            prefecture = values.get("prefecture")
            if prefecture not in cls.JAPAN_PREFERCTURES:
                raise ValueError("日本の都道府県を指定してください。")
            return values

    # Set up a parser + inject instructions into the prompt template.
    parser = PydanticOutputParser(pydantic_object=SightSeeingSpotsInJapan)

    print(parser.get_format_instructions())
    """
    The output should be formatted as a JSON instance that conforms to the JSON schema below.

    As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
    the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

    Here is the output schema:
    ```
    {"properties": {"prefecture": {"description": "旅行先の県名", "title": "Prefecture", "type": "string"}, "spots": {"description": "その県の有名な観光地", "items": {"type": "string"}, "title": "Spots", "type": "array"}}, "required": ["prefecture", "spots"]}
    ```
    """

    prompt = PromptTemplate(
        template="ユーザの質問に答えなさい.\n{format_instructions}\n\nquery: {query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # And a query intended to prompt a language model to populate the data structure.
    prompt_and_model = prompt | model
    output = prompt_and_model.invoke({"query": "茨城県の有名な観光地を教えて"})
    print(type(output)) # str
    print(output)
    """
    {
    "prefecture": "茨城県",
    "spots": [
        "日立市の常陸太田市",
        "水戸市の偕楽園",
        "つくば市の筑波山",
        "鹿島市の鹿島神宮",
        "ひたちなか市のひたち海浜公園"
    ]
    }
    """
    response = parser.invoke(output)
    print(type(response))
    # <class '__main__.main_pydantic_output_parser.<locals>.SightSeeingSpotsInJapan'>

    print(response)
    # prefecture='茨城県' spots=['日立市の常陸太田市', '水戸市の偕楽園', 'つくば市の筑波山', '鹿島市の鹿島神宮', 'ひたちなか市のひたち海浜公園']

    # And a query intended to prompt a language model to populate the data structure.
    chain = prompt | model | parser
    response = chain.invoke({"query": "茨城県の有名な観光地を教えて"})
    print(response)
    # prefecture='茨城県' spots=['日立市の常陸太田市', '水戸市の偕楽園', 'つくば市の筑波山', '鹿島市の鹿島神宮', 'ひたちなか市のひたち海浜公園']

    for chunk in chain.stream({"query": "北海道の有名な観光地を教えて"}):
        print(chunk, end="\n", flush=True) # flush=True は出力をバッファにためないための措置
    
    """
    prefecture='北海道' spots=['']
    prefecture='北海道' spots=['旭']
    prefecture='北海道' spots=['旭山']
    prefecture='北海道' spots=['旭山動']
    prefecture='北海道' spots=['旭山動物']
    prefecture='北海道' spots=['旭山動物園']
    prefecture='北海道' spots=['旭山動物園', '']
    prefecture='北海道' spots=['旭山動物園', '函']
    prefecture='北海道' spots=['旭山動物園', '函館']
    prefecture='北海道' spots=['旭山動物園', '函館山']
    prefecture='北海道' spots=['旭山動物園', '函館山', '']
    prefecture='北海道' spots=['旭山動物園', '函館山', '洞']
    prefecture='北海道' spots=['旭山動物園', '函館山', '洞爺']
    prefecture='北海道' spots=['旭山動物園', '函館山', '洞爺湖']
    """

    # # 都道府県名が不適なので validation error が発生する
    # response = chain.invoke({"query": "横浜県の有名な観光地を教えて"})

def main_json_output_parser():
    from langchain.output_parsers.json import SimpleJsonOutputParser

    model = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.0)
    json_prompt = PromptTemplate.from_template(
        "Return a JSON object with an `answer` key that answers the following question: {question}"
    )
    json_parser = SimpleJsonOutputParser()
    chain = json_prompt | model | json_parser
    response = (json_prompt | model).invoke({"question": "Langchainの名前の由来を教えてください"})
    print(type(response)) # str

    response = chain.invoke({"question": "Langchainの名前の由来を教えてください"})
    print(type(response)) # dict
    print(response)
    

if __name__ == "__main__":
    main_pydantic_output_parser()
    main_json_output_parser()