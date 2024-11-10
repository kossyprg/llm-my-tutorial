from dotenv import load_dotenv
load_dotenv()
from langfuse.decorators import observe
from langfuse.openai import openai # OpenAI integration

@observe(name="大喜利ジェネレーション")
def story():
    return openai.chat.completions.create(
        model="gpt-4o",
        max_tokens=150,
        temperature=0.8,
        messages=[
          {"role": "system", "content": "あなたはユーモアのセンスのある芸人です。ユーザの出すお題に対して洒落の利いた面白い回答をしてください。"},
          {"role": "user", "content": "こんな大規模言語モデル(Large Language Model: LLM)は嫌だ。どんなLLM?"},
          {"role": "assistant", "content": "夏になって気温(temperature)が高くなるとでたらめなことを言う。"},
          {"role": "user", "content": "こんな大規模言語モデル(Large Language Model: LLM)は嫌だ。どんなLLM?"},
        ],
    ).choices[0].message.content
 
@observe(name="デコレータ使用例")
def decorator_example():
    return story()

if __name__ == "__main__":
    # @observe() を使う方法
    # Ref https://langfuse.com/docs/get-started#log-your-first-llm-call-to-langfuse
    print(decorator_example())
