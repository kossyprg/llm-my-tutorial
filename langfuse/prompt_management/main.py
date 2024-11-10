from dotenv import load_dotenv
load_dotenv()

import os
from langfuse import Langfuse
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.prompts import PromptTemplate    # from_template()に直接メタデータを付与できるが、一般に ChatPromptTemplate の方が使い勝手がいい？
from langchain_core.output_parsers.string import StrOutputParser
from langchain_openai import ChatOpenAI
from langfuse.callback import CallbackHandler

# Ref https://langfuse.com/docs/prompts/get-started
def fetch_prompt(
    name: str, 
    label: str = None, 
) -> ChatPromptTemplate:
    # Initialize Langfuse client
    langfuse = Langfuse()
    
    # version または label を指定しない場合、production ラベルが張られているものが選択される。
    # もし version と label の両方とも指定せず、production ラベルが張られているものがなければ LangfuseNotFoundError になる。
    langfuse_prompt = langfuse.get_prompt(name=name, label=label)
    
    # Example using ChatPromptTemplate
    # get_langchain_prompt は langfuse の prompt を langchain に適合するためのメソッド
    # 二重カッコ {{}} を一重カッコ {} に変換している。二重カッコで指定する方法を mustache-style という。詳細は mustache.js で検索。
    langchain_prompt = ChatPromptTemplate.from_template(
        template=langfuse_prompt.get_langchain_prompt(), 
    )

    # `langfuse_prompt` のメタデータは実行結果と使用したプロンプトのバージョンを履歴に残すために必要
    # langfuse_prompt のメタデータはPromptTemplateのみに設定し、LLMや他のchainに設定しないこと
    # Ref https://langfuse.com/docs/prompts/get-started#link-with-langfuse-tracing-optional
    langchain_prompt.metadata = {"langfuse_prompt": langfuse_prompt}
    
    # Example using ChatPromptTemplate with pre-compiled variables.
    # プロンプトに `question` という変数があって、予め値を入れたい場合はそれを指定できる。
    # langchain_prompt = ChatPromptTemplate.from_template(langfuse_prompt.get_langchain_prompt(question='foo'))
    # print(langchain_prompt)

    return langchain_prompt

def main():
    prompt = fetch_prompt(name="my-rag-prompt")
    print(type(prompt)) # langchain_core.prompts.chat.ChatPromptTemplate

    # Ref https://x.com/NintendoCoLtd/status/1853972161238847712
    # 閲覧日: 2024/11/10
    context = "古川です。本日の経営方針説明会で、Nintendo Switchの後継機種ではNintendo Switch向けソフトも遊べることを公表しました。また、Nintendo Switch Onlineも後継機種で引き続きご利用いただけるようにします。これらNintendo Switchとの互換性を含む後継機種の詳しい情報は、後日改めてご案内します。"

    question = "Nintendo Switchの後継機種でNintendo Switch向けソフトを遊ぶことはできますか?"

    llm = ChatOpenAI(model="gpt-4o")
    chain = prompt | llm | StrOutputParser()

    langfuse_handler = CallbackHandler(
        secret_key=os.environ['LANGFUSE_SECRET_KEY'],
        public_key=os.environ['LANGFUSE_PUBLIC_KEY'],
        host=os.environ['LANGFUSE_HOST'],
    )

    result = chain.invoke({"question": question, "context": context}, 
                          config={"callbacks": [langfuse_handler], "run_name": "RAG"})
    print(result)

if __name__ == "__main__":
    main()



