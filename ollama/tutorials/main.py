from dotenv import load_dotenv
load_dotenv()
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
# API reference: https://python.langchain.com/docs/integrations/chat/ollama/

def main():
    model_name = "llama3.2:3b"

    llm = ChatOllama(
        model=model_name,
        temperature=0.2,
        base_url="http://host.docker.internal:11434",
        seed=0 # 結果を再現したいときは乱数の種を指定する
        # other params...
    )

    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to Japanese. Translate the user sentence. ",
        ),
        ("human", "Time flies."), # 光陰矢の如し
    ]
    chain = llm | StrOutputParser()
    ai_msg = chain.invoke(messages, {"run_name": f"ChatOllama ({model_name})"})
    print(ai_msg) # "（時は飛びます。）" カッコが付く理由は不明。

def main_use_chat_prompt_template():
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that translates {input_language} to {output_language}.Translate the user sentence.",
            ),
            ("human", "{input}"),
        ]
    )

    model_name = "llama3.2:3b"
    llm = ChatOllama(
        model=model_name,
        temperature=0.2,
        base_url="http://host.docker.internal:11434",
        seed=4 # 結果を再現したいときは乱数の種を指定する
        # other params...
    )

    # 乱数の種を指定しても結果が再現しない現象を確認(langchain-ollama==0.2.0). 
    # ダミーのinvoke()を実施すれば再現する。
    # 関連と思われるIssue
    # https://github.com/langchain-ai/langchain/issues/24703
    # https://github.com/rasbt/LLMs-from-scratch/issues/249
    llm.invoke("Do nothing.", {"run_name": "Do nothing", "tags": ["dummy"]}) 

    chain = prompt | llm | StrOutputParser()
    res = chain.invoke(
        {
            "input_language": "German",
            "output_language": "Japanese",
            "input": "Hören Sie bitte", # 聞いてください(Please listen)
        },
        {"run_name": f"ChatOllama with prompt template ({model_name})"}
    )
    print(res) # "お聞きください"

if __name__ == "__main__":
    main()
    main_use_chat_prompt_template()