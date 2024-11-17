from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith.schemas import Example, Run
from langsmith.evaluation import evaluate

def main():
    system_prompt = "あなたは、日本語の文章を感情的な性質に基づいて分類する専門家です。以下のルールに従って、ユーザのクエリを「ポジティブ」または「ネガティブ」に分類してください。出力は「ポジティブ」または「ネガティブ」のみ許される。\n\n" + \
                "### 分類ルール\n" + \
                "1. **ポジティブ**:\n" + \
                "   - 内容が明るく、前向きな感情を表している場合。\n" + \
                "   - 喜び、満足、感謝、期待などを含む文章。\n" + \
                "   - 例: 「今日は素晴らしい一日だった！」「新しい挑戦が楽しみです」「あなたの努力に感謝しています」\n\n" + \
                "2. **ネガティブ**:\n" + \
                "   - 内容が暗く、後ろ向きな感情を表している場合。\n" + \
                "   - 悲しみ、不満、諦め、怒りなどを含む文章。\n" + \
                "   - 例: 「もう何もかもうまくいかない」「最悪の結果になった」「どうしてこんなに失敗ばかりなんだろう」\n\n" + \
                "### 注意事項\n" + \
                "- 文脈を正確に読み取り、適切な分類をしてください。\n" + \
                "- どちらの分類にも該当しない場合は、最も近い性質を選んでください。\n" + \
                "- 中立的な表現が含まれる場合でも、全体のトーンを考慮して分類してください。\n\n"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{text}") # データセット作成時の入力のキー名と一致しないとエラー。
    ])
    
    chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    output_parser = StrOutputParser()
    chain = prompt | chat_model | output_parser
    
    dataset_name = "Positive or negative dataset"

    def correct_label(root_run: Run, example: Example) -> dict:
        score = root_run.outputs.get("output") == example.outputs.get("label")
        return {"score": int(score), "key": "correct_label"}
    
    results = evaluate(
        chain.invoke, # chain.invoke を渡す
        data=dataset_name,
        evaluators=[correct_label],
        experiment_prefix="Positive or negative (langchain runnable)",
        description="langchainのrunnableを使ってテストを実施しています。",
    )

if __name__ == "__main__":
    main()
