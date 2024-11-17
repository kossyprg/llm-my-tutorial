from dotenv import load_dotenv
load_dotenv()

from langsmith import traceable, wrappers
import openai
from langsmith import Client
from langsmith.schemas import Example, Run

def main():
    # ステップ1 : テキストがポジティブかネガティブかを判定するタスクを定義
    @traceable
    def label_text(text):
        # wrap_openai はいちいち traceable としなくてもトレースしてくれるラッパーらしい。
        # Ref https://docs.smith.langchain.com/old/tracing/faq/logging_and_viewing#wrapping-the-openai-client
        client_openai = wrappers.wrap_openai(openai.Client())

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
        
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": text},
        ]
        
        result = client_openai.chat.completions.create(
            messages=messages, model="gpt-3.5-turbo-0125", temperature=0
        )
        return result.choices[0].message.content

    # ステップ2 : データセットを作成あるいは読み込み
    client_langsmith = Client()
    
    # Create a dataset
    examples = [
        ("このケーキ、マジで美味しい！", "ポジティブ"),
        ("あの映画は観るだけ時間の無駄だ。", "ネガティブ"),
        ("今日も推しが尊い", "ポジティブ"),
        ("お前の料理、誰も食べたくないってさ", "ネガティブ"),
        ("TOEIC俺は最高で942点を取りましたけど?", "ポジティブ"),
        ("AIプロジェクトでPoCを作ったんだけど、誰も使ってくれない...", "ネガティブ"),
    ]

    # 既にこのデータセットがある場合は 409 Client Error になるので、ないときだけ作成。
    dataset_name = "Positive or negative dataset"
    if not client_langsmith.has_dataset(dataset_name=dataset_name):
        print("Create dataset")
        dataset = client_langsmith.create_dataset(dataset_name=dataset_name)
        inputs, outputs = zip(
            *[({"text": text}, {"label": label}) for text, label in examples]
        )
        client_langsmith.create_examples(inputs=inputs, outputs=outputs, dataset_id=dataset.id)

    # ステップ3 : スコアを出力するための evaluator を構成
    def correct_label(root_run: Run, example: Example) -> dict:
        score = root_run.outputs.get("output") == example.outputs.get("label")
        return {"score": int(score), "key": "correct_label"}

    # ステップ4 : 評価を実行、結果を調査
    from langsmith.evaluation import evaluate

    results = evaluate(
        lambda inputs: label_text(inputs["text"]),
        data=dataset_name,
        evaluators=[correct_label],
        experiment_prefix="Positive or negative",
        description="基本的なevaluationの流れを確認するためのテストです。",  # optional
    )

if __name__ == "__main__":
    main()
