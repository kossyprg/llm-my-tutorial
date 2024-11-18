from dotenv import load_dotenv
load_dotenv()


from langchain_openai import ChatOpenAI
from langsmith.evaluation import evaluate

# 以下は自作したモジュール
from embeddings import load_hf_embeddings
from vectorstore import setup_vectorstore
from retriever import MelosRetriever
from tool import MelosQATool, MelosQATool_SingleOutput
from dataset import create_dataset_if_not_exist, DATASET_NAME
from utils import print_dict_contents

from evaluators import answer_evaluator
from evaluators import answer_helpfulness_evaluator
from evaluators import answer_hallucination_evaluator
from evaluators import docs_relevance_evaluator
from evaluators import document_relevance_grader
from evaluators import answer_hallucination_grader


def main():
    embeddings = load_hf_embeddings() # 埋め込みモデルの用意
    vectorstore = setup_vectorstore(embeddings) # ベクトルデータベースの作成
    retriever = MelosRetriever(vectorstore=vectorstore, k=4, name="走れメロス検索エンジン") # 検索エンジン
    tool = MelosQATool(retriever=retriever) # RAGアーキテクチャを備えたツール

    # # 単に実行してみたいときは以下
    # result = tool.invoke("メロスは太陽の沈む速度の何倍で走りましたか？")
    # print_dict_contents(result)

    # データセットの作成
    create_dataset_if_not_exist()

    # targetの定義
    # データセット(example)を受け取り、応答およびコンテキストを返す
    def predict_rag_answer(example: dict):
        """Use this for answer evaluation"""
        response = tool.invoke(example["question"]) # データセットのキー名と一致させないとエラー
        return {"answer": response["answer"]}

    def predict_rag_answer_with_context(example: dict):
        """Use this for evaluation of retrieved documents and hallucinations"""
        response = tool.invoke(example["question"])
        return {"answer": response["answer"], "contexts": response["contexts"]}
    
    # 1. 生成した応答 vs 正解
    evaluate(
        predict_rag_answer,
        data=DATASET_NAME,
        evaluators=[answer_evaluator],
        experiment_prefix="rag-answer-v-reference",
        metadata={"version": "LCEL context, gpt-4-0125-preview"},
    )

    # 2. 生成した応答 vs 質問文
    evaluate(
        predict_rag_answer,
        data=DATASET_NAME,
        evaluators=[answer_helpfulness_evaluator],
        experiment_prefix="rag-answer-helpfulness",
        metadata={"version": "LCEL context, gpt-4-0125-preview"},
    )

    # 3. 生成した応答 vs 取得したチャンク
    evaluate(
        predict_rag_answer_with_context,
        data=DATASET_NAME,
        evaluators=[answer_hallucination_evaluator],
        experiment_prefix="rag-answer-hallucination",
        metadata={"version": "LCEL context, gpt-4-0125-preview"},
    )

    # 4. 取得したチャンク vs 質問文
    evaluate(
        predict_rag_answer_with_context,
        data=DATASET_NAME,
        evaluators=[docs_relevance_evaluator],
        experiment_prefix="rag-doc-relevance",
        metadata={"version": "LCEL context, gpt-4-0125-preview"},
    )

# MelosQATool は context を tool.invoke() の戻り値に追加している
# 以下では、evaluator 内で所望の Run を取得することで、
# invoke() の戻り値を変更せずともテストを実行できるようにしている
def main_single_output_chain():
    embeddings = load_hf_embeddings() # 埋め込みモデルの用意
    vectorstore = setup_vectorstore(embeddings) # ベクトルデータベースの作成
    retriever = MelosRetriever(vectorstore=vectorstore, k=4, name="走れメロス検索エンジン") # 検索エンジン
    tool = MelosQATool_SingleOutput(retriever=retriever) # RAGアーキテクチャを備えたツール

    def predict_rag_answer(example: dict):
        """Use this for answer evaluation"""
        response = tool.invoke(example["question"]) # データセットのキー名と一致させないとエラー
        return {"answer": response}
    
    evaluate(
        predict_rag_answer,
        data=DATASET_NAME,
        evaluators=[document_relevance_grader, answer_hallucination_grader],
        experiment_prefix="rag-without-IF-changes",
        metadata={"version": "LCEL context, gpt-4-0125-preview"},
    )

if __name__ == "__main__":
    main()
    main_single_output_chain()
    