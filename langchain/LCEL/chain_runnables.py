# Ref https://python.langchain.com/docs/how_to/sequence/
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith.schemas import Example, Run
from langsmith.evaluation import evaluate

def main():
    model = ChatOpenAI(model="gpt-4o-mini")

    system_prompt = "あなたはしりとりゲームを行うAIアシスタントです。" + \
                    "最初の単語の最後の文字を次の単語の最初に使います。" + \
                    "単語は日本語で、「ん」で終わる単語を言った場合、負けになります。" + \
                    "回答はひらがなのみ許可され、漢字を使用してはいけません。" + \
                    "また、単語以外の回答やルール違反をしないでください。"
    
    prompt_template = ChatPromptTemplate([
        ("system", system_prompt),
        ("user", "{input}")
    ])

    # パイプ演算子 | を使って2つ以上の Runnable オブジェクトを連結する
    chain = prompt_template | model | StrOutputParser()
    response = chain.invoke({"input": "りんご"}, config={"run_name": "simple chain"})
    print(response) # ごま

    judge_prompt = ChatPromptTemplate([
        ("system", "与えられた単語が「ん」で終わっているかどうか判定しなさい"),
        ("user", "{input}")
    ])

    # chain を組み合わせることもできます。次のチェーンへの入力のフォーマットに注意が必要です。
    composed_chain = {"input": chain} | judge_prompt | model | StrOutputParser()

    response = composed_chain.invoke({"input": "りんご"}, config={"run_name": "composed_chain"})
    print(response) # 「ごま」は「ん」で終わっていません。

    # lambda で定義した関数を連結すると、Runnable に変換されます。
    # ただし、streaming が効かなくなることに注意が必要です。
    # streaming が必要な場合は RunnableGenerator を使います。
    # Ref https://python.langchain.com/docs/how_to/functions/#streaming
    composed_chain_with_lambda = (
        chain
        | (lambda input: {"input": input})
        | judge_prompt
        | model
        | StrOutputParser()
    )

    response = composed_chain_with_lambda.invoke({"input": "りんご"}, config={"run_name": "composed_chain_with_lambda"})
    print(response) # 「ごりら」は「ん」で終わっていません。

    # pipe() メソッドを使っても連結可能です。
    chain = prompt_template.pipe(model).pipe(StrOutputParser())
    # chain = prompt_template | model | StrOutputParser() と同じ
    response = chain.invoke({"input": "りんご"}, config={"run_name": "simple chain with pipe()"})
    print(response) # ごま

if __name__ == "__main__":
    main()