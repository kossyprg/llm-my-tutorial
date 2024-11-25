# Ref https://python.langchain.com/docs/how_to/few_shot_examples/
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

example_prompt = PromptTemplate.from_template("質問: {question}\n{answer}")

# Ref https://bocek.co.jp/media/exercise/prompt-engineer-exercise/3087/
examples = [
    {
        "question": "「こんにちは」を逆から読むと？",
        "answer": """
ステップ1: 先頭から順に番号を振ったリストを生成します。
[(0, 'こ'), (1, 'ん'), (2, 'に'), (3, 'ち'), (4, 'は')]
ステップ2: 値の大きいものから順に取得してリストを作成します。
['は', 'ち', 'に', 'ん', 'こ']
ステップ3: ステップ2で得たリストを連結した文字列を出力します。
'はちにんこ'
よって、最終的な回答は「はちにんこ」です。
"""
    },
    {
        "question": "「さくら」を逆から読むと？",
        "answer": """
ステップ1: 先頭から順に番号を振ったリストを生成します。
[(0, 'さ'), (1, 'く'), (2, 'ら')]
ステップ2: 値の大きいものから順に取得してリストを作成します。
['ら', 'く', 'さ']
ステップ3: ステップ2で得たリストを連結した文字列を出力します。
'らくさ'
よって、最終的な回答は「らくさ」です。
"""
    },
    {
        "question": "「ありがとう」を逆から読むと？",
        "answer": """
ステップ1: 先頭から順に番号を振ったリストを生成します。
[(0, 'あ'), (1, 'り'), (2, 'が'), (3, 'と'), (4, 'う')]
ステップ2: 値の大きいものから順に取得してリストを作成します。
['う', 'と', 'が', 'り', 'あ']
ステップ3: ステップ2で得たリストを連結した文字列を出力します。
'うとがりあ'
よって、最終的な回答は「うとがりあ」です。
"""
    },
    {
        "question": "次のグループの奇数を合計すると偶数になる。YesかNoか。[1、3、6、8、9、11、12]",
        "answer": """
ステップ1: グループの中に含まれる奇数が偶数個であれば奇数の合計は偶数、奇数個であれば奇数であることに気付く。
ステップ2: グループの中に含まれる奇数が何個あるか順に調べる。
奇数の数を odd_num として odd_num = 0 で初期化する。
1は奇数なのでodd_num = odd_num + 1 を実行して odd_num = 1 となる
3は奇数なのでodd_num = odd_num + 1 を実行して odd_num = 2 となる
6は偶数なのでodd_num = 2 のまま。
8は偶数なのでodd_num = 2 のまま。
9は奇数なのでodd_num = odd_num + 1 を実行して odd_num = 3 となる
11は奇数なのでodd_num = odd_num + 1 を実行して odd_num = 4 となる
12は偶数なのでodd_num = 4 のまま。
ステップ3: ステップ2で得た奇数の個数とステップ1の考え方より、奇数の個数が偶数個であるので、与えられたグループの奇数の和は偶数である。
よって、最終的な回答は「Yes」です。
"""
    },
    {
        "question": "次のグループの奇数を合計すると偶数になる。YesかNoか。[4、6、9、11、14、17、20]",
        "answer": """
ステップ1: グループの中に含まれる奇数が偶数個であれば奇数の合計は偶数、奇数個であれば奇数であることに気付く。
ステップ2: グループの中に含まれる奇数が何個あるか順に調べる。
奇数の数を odd_num として odd_num = 0 で初期化する。
4は偶数なのでodd_num = 0 のまま。
6は偶数なのでodd_num = 0 のまま。
9は奇数なのでodd_num = odd_num + 1 を実行して odd_num = 1 となる
11は奇数なのでodd_num = odd_num + 1 を実行して odd_num = 2 となる
14は偶数なのでodd_num = 2 のまま。
17は奇数なのでodd_num = odd_num + 1 を実行して odd_num = 3 となる
20は偶数なのでodd_num = 3 のまま。
ステップ3: ステップ2で得た奇数の個数とステップ1の考え方より、奇数の個数が奇数個であるので、与えられたグループの奇数の和は奇数である。
よって、最終的な回答は「No」です。
"""
    }
]

def main_few_shot_examples():
    from langchain_core.prompts import FewShotPromptTemplate

    # 例を一つ入れてフォーマットした例
    print(example_prompt.invoke(examples[0]).to_string())
    """
    質問: 「こんにちは」を逆から読むと？

    ステップ1: 先頭から順に番号を振ったリストを生成します。
    [(0, 'こ'), (1, 'ん'), (2, 'に'), (3, 'ち'), (4, 'は')]
    ステップ2: 値の大きいものから順に取得してリストを作成します。
    ['は', 'ち', 'に', 'ん', 'こ']
    ステップ3: ステップ2で得たリストを連結した文字列を出力します。
    'はちにんこ'
    よって、最終的な回答は「はちにんこ」です。
    """

    prompt = FewShotPromptTemplate(
        examples=examples, # 例を格納したリスト
        example_prompt=example_prompt, # 各shotをフォーマットするためのプロンプト
        suffix="質問: {input}", # 例の後に入れる文字列
        input_variables=["input"],
    )

    # 例の後に入力した質問が挿入される
    # print(prompt.invoke({"input": "「はじめまして」を逆から読むと？"}).to_string())

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    chain = prompt | llm
    response = chain.invoke({"input": "「はじめまして」を逆から読むと？"})
    print(response.content)

    response = chain.invoke({"input": "次のグループの奇数を合計すると偶数になる。YesかNoか。[3, 1, 4, 1, 5, 9, 2]"})
    print(response.content)

def main_few_shot_examples_with_selector():
    from langchain_chroma import Chroma
    from langchain_core.example_selectors import SemanticSimilarityExampleSelector
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.prompts import FewShotPromptTemplate

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        # This is the list of examples available to select from.
        examples,
        # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
        OpenAIEmbeddings(),
        # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
        Chroma,
        # This is the number of examples to produce.
        k=1,
        persist_directory="./questions_db" # **vectorstore_cls_kwargs に追加
    )

    # Select the most similar example to the input.
    question = "「かるいきびんなこねこ」を逆から読むと？" # Ref https://blog.nazo2.net/4200/
    selected_examples = example_selector.select_examples({"question": question})
    print(f"Examples most similar to the input: {question}")
    for example in selected_examples:
        print("\n")
        for k, v in example.items():
            print(f"{k}: {v}")
    
    prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        suffix="Question: {input}",
        input_variables=["input"],
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    chain = prompt | llm

    question = "「かるいきびんなこねこ」を逆から読むと？"
    response = chain.invoke({"input": question})
    print(response.content)

    question = "次のグループの奇数を合計すると偶数になる。YesかNoか。[2, 2, 3, 6, 0, 6, 7, 9]"
    response = chain.invoke({"input": question})
    print(response.content)

if __name__ == "__main__":
    main_few_shot_examples()
    main_few_shot_examples_with_selector()