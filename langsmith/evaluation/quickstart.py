from dotenv import load_dotenv
load_dotenv()
from langsmith import evaluate, Client
from langsmith.schemas import Example, Run

# 1. Create and/or select your dataset
client = Client()
dataset = client.clone_public_dataset("https://smith.langchain.com/public/a63525f9-bdf2-4512-83e3-077dc9417f96/d")
# name='ds-respectful-meteorology-85' 

# 2. Define an evaluator
# For more info on defining evaluators, see: https://docs.smith.langchain.com/evaluation/how_to_guides/evaluation/evaluate_llm_application#use-custom-evaluators
def is_concise_enough(root_run: Run, example: Example) -> dict:
    print(root_run.outputs['output']) # What do mammals and birds have in common? is a good question. I don't know the answer.
    print(example.outputs['answer'])  # They are both warm-blooded
    score = 10
    return {"key": "dummy_key", "score": score}

def hoge_evaluator(root_run: Run, example: Example) -> dict:
    return {"key": "hoge", "score": 3.14}

# 複数のスコアを返すこともできる
# https://docs.smith.langchain.com/evaluation/how_to_guides/evaluation/evaluate_llm_application#return-multiple-scores
def multiple_scores(root_run: Run, example: Example) -> dict:
    return {
        "results": [
            {"key": "precision", "score": 0.8},
            {"key": "recall", "score": 0.9},
            {"key": "f1", "score": 0.85},
        ]
    }

# 3. Run an evaluation
evaluate(
    lambda x: x["question"] + " is a good question. I don't know the answer.",
    data=dataset.name,
    evaluators=[is_concise_enough, hoge_evaluator, multiple_scores],
    experiment_prefix="langsmith-evaluation-tutorial experiment"
)