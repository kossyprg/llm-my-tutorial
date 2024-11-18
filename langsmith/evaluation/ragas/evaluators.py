from langchain import hub
from langchain_openai import ChatOpenAI
from langsmith.schemas import Example, Run
from typing import List
from utils import print_dict_contents

############################################################
# 1. 生成した応答 vs 正解
############################################################
def answer_evaluator(run: Run, example: Example) -> dict:
    """
    生成した応答と正解を比較して0~1で採点する。

    ### 採点基準(詳細はhubのプロンプトを参照)
    1. 生成した応答が事実として正確か
    2. 生成した応答に矛盾する内容が含まれていないか
    3. 生成した応答に正解よりも多くの情報を含んでいてもよい。ただし、それが事実として正確であり、正解と矛盾しない場合に限る。
    """

    # Grade prompt
    # Ref https://smith.langchain.com/hub/langchain-ai/rag-answer-vs-reference
    grade_prompt_answer_accuracy = hub.pull("langchain-ai/rag-answer-vs-reference")

    # Get question, ground truth answer, RAG chain answer
    input_question = example.inputs["question"] # ユーザのクエリ
    reference = example.outputs["answer"]       # 正解
    prediction = run.outputs["answer"]          # 生成した応答

    # LLM grader
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

    # Structured prompt
    answer_grader = grade_prompt_answer_accuracy | llm

    # Run evaluator
    score = answer_grader.invoke({"question": input_question,    # QUESTION
                                  "correct_answer": reference,   # GROUND_TRUTH_ANSWER
                                  "student_answer": prediction}) # STUDENT_ANSWER
    score = score["Score"]

    return {"key": "answer_v_reference_score", "score": score}


############################################################
# 2. 生成した応答 vs 質問文
############################################################
def answer_helpfulness_evaluator(run: Run, example: Example) -> dict:
    """
    生成した応答が質問文に対する回答として役に立つか0~1で採点する。

    ### 採点基準(詳細はhubのプロンプトを参照)
    1. 生成した応答が質問に対して簡潔かつ適切である
    2. 生成した応答が質問に対する答えとして役に立つこと
    """
    # Grade prompt
    # Ref https://smith.langchain.com/hub/langchain-ai/rag-answer-helpfulness
    grade_prompt_answer_helpfulness = hub.pull("langchain-ai/rag-answer-helpfulness")

    # Get question, ground truth answer, RAG chain answer
    input_question = example.inputs["question"] # キー名に注意
    prediction = run.outputs["answer"]

    # LLM grader
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

    # Structured prompt
    answer_grader = grade_prompt_answer_helpfulness | llm

    # Run evaluator
    score = answer_grader.invoke({"question": input_question,
                                  "student_answer": prediction})
    score = score["Score"]

    return {"key": "answer_helpfulness_score", "score": score}

############################################################
# 3. 生成した応答 vs 取得したチャンク
############################################################
def answer_hallucination_evaluator(run: Run, example: Example) -> dict:
    """
    事実に基づいた回答になっているかを0~1で採点する。

    ### 採点基準(詳細はhubのプロンプトを参照)
    1. 生成した応答がコンテキストに基づいていること
    2. 生成した応答に、コンテキストの範囲外の情報が含まれていないこと
    """
    # Grade prompt
    # Ref https://smith.langchain.com/hub/langchain-ai/rag-answer-hallucination
    grade_prompt_hallucinations = hub.pull("langchain-ai/rag-answer-hallucination")

    # RAG inputs
    # input_question = example.inputs["question"] # 使わない
    contexts = run.outputs["contexts"]

    # RAG answer
    prediction = run.outputs["answer"]

    # LLM grader
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

    # Structured prompt
    answer_grader = grade_prompt_hallucinations | llm

    # Get score
    score = answer_grader.invoke({"documents": contexts,
                                  "student_answer": prediction})
    score = score["Score"]

    return {"key": "answer_hallucination", "score": score}

############################################################
# 4. 取得したチャンク vs 質問文
############################################################
def docs_relevance_evaluator(run: Run, example: Example) -> dict:
    """
    質問文とコンテキストの関連性を0~1で採点する。

    ### 採点基準(詳細はhubのプロンプトを参照)
    1. 質問と無関係な事実を特定することが目標。
    2. コンテキストに質問と関連するキーワードや意味的な関連性が含まれる場合、それを関連があるとみなす。
    3. 2が満たされるなら、コンテキストの中に質問と無関係な情報が一部含まれていても問題ない。
    """
    # Grade prompt
    # Ref https://smith.langchain.com/hub/langchain-ai/rag-document-relevance
    grade_prompt_doc_relevance = hub.pull("langchain-ai/rag-document-relevance")

    # RAG inputs
    input_question = example.inputs["question"]
    contexts = run.outputs["contexts"]

    # LLM grader
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

    # Structured prompt
    answer_grader = grade_prompt_doc_relevance | llm

    # Get score
    score = answer_grader.invoke({"question": input_question,
                                  "documents": contexts})
    score = score["Score"]

    return {"key": "document_relevance", "score": score}

############################################################
# インタフェースを変更しなくても、所望の Run を検索すれば同様の試験が可能
# RunExplorer は所望の Run を検索するモジュール
# Run の名前が一意に決まらないといけないことに注意
############################################################
class RunExplorer:
    @staticmethod
    def get_target_run(root_run: Run, target_run_name: str):
        """
        深さ優先探索で特定の名前の Run オブジェクトを探す。
        制約: 目標となる Run の名前と同名の Run が存在しないこと

        Args:
            root_run (Run): 探索の起点となるルートの Run オブジェクト。
            target_run_name (str): 探したい Run の名前。
        
        Returns:
            Run: 見つかった場合は対象の Run オブジェクト、見つからなければ None。
        """
        # 初期化された空の訪問リストを用意して DFS を開始
        return RunExplorer._dfs(root_run, target_run_name, [])
    
    @staticmethod
    def _dfs(run: Run, target_run_name: str, visited: List) -> Run:
        """
        深さ優先探索 (DFS) で指定された名前の Run を探す。
        
        Args:
            run (Run): 現在探索中の Run オブジェクト。
            target_run_name (str): 探したい Run の名前。
            visited (List): 訪問済みの Run 名のリスト。
        
        Returns:
            Run: 見つかった場合は対象の Run オブジェクト、見つからなければ None。
        """
        if run.name == target_run_name:
            return run
        
        if run.name in visited:
            return None
        
        visited.append(run.name)
        
        # 子の Run オブジェクトを再帰的に探索
        for run_child in run.child_runs:
            result = RunExplorer._dfs(run_child, target_run_name, visited)
            # 見つかった場合、その Run を返す
            if result is not None:
                return result
        
        # 子ノード全体を探索しても見つからなかった場合 None を返す
        return None

def document_relevance_grader(root_run: Run, example: Example) -> dict:
    """
    A simple evaluator that checks to see if retrieved documents are relevant to the question
    """
    grade_prompt_doc_relevance = hub.pull("langchain-ai/rag-document-relevance")

    # コンテキストを出力する Run を取得
    context_run = RunExplorer.get_target_run(root_run, "コンテキスト抽出")
    contexts = context_run.outputs["output"]

    input_question = example.inputs["question"]

    # LLM grader
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

    # Structured prompt
    answer_grader = grade_prompt_doc_relevance | llm

    # Get score
    score = answer_grader.invoke({"question":input_question,
                                "documents":contexts})
    score = score["Score"]

    return {"key": "document_relevance", "score": score}

def answer_hallucination_grader(root_run: Run, example: Example) -> dict:
    """
    A simple evaluator that checks to see the answer is grounded in the documents
    """
    grade_prompt_hallucinations = hub.pull("langchain-ai/rag-answer-hallucination")

    # コンテキストを出力する Run を取得
    context_run = RunExplorer.get_target_run(root_run, "コンテキスト抽出")
    contexts = context_run.outputs["output"]

    # 最終出力を行う Run を取得
    tool_run = RunExplorer.get_target_run(root_run, "走れメロス質疑応答ツール")
    prediction = tool_run.outputs["output"]

    # LLM grader
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

    # Structured prompt
    answer_grader = grade_prompt_hallucinations | llm

    # Get score
    score = answer_grader.invoke({"documents": contexts,
                                  "student_answer": prediction})
    score = score["Score"]

    return {"key": "answer_hallucination", "score": score}