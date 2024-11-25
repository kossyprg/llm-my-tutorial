# Ref https://python.langchain.com/docs/concepts/prompt_templates/
from dotenv import load_dotenv
load_dotenv()

def main_string_prompt_templates():
    from langchain_core.prompts import PromptTemplate

    # 単一の文字列をフォーマットする
    prompt_template = PromptTemplate.from_template("{topic}についてジョークを言って")

    prompt_value = prompt_template.invoke({"topic": "猫"}, config={"run_name": "String prompt templates"})

    # PromptValue は LLM や ChatModel に渡すために使われる。
    # 文字列と Message の切り替えを容易にするために使われるらしい。
    print(type(prompt_value)) # <class 'langchain_core.prompt_values.StringPromptValue'>
    print(prompt_value)       # text='猫についてジョークを言って'

def main_chat_prompt_templates():
    from langchain_core.prompts import ChatPromptTemplate

    prompt_template = ChatPromptTemplate([
        ("system", "あなたは優秀なAIアシスタントです。"),
        ("user", "{topic}についてジョークを言って")
    ])

    prompt_value = prompt_template.invoke({"topic": "猫"}, config={"run_name": "Simple ChatPromptTemplate"})
    print(type(prompt_value)) # <class 'langchain_core.prompt_values.ChatPromptValue'>
    print(prompt_value)       
    # messages=[SystemMessage(content='あなたは優秀なAIアシスタントです。', additional_kwargs={}, response_metadata={}), 
    # HumanMessage(content='猫についてジョークを言って', additional_kwargs={}, response_metadata={})]

def main_messages_placeholder():
    # MessagePlaceHolderは特定の場所にメッセージのリストを挿入するときに使う
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import HumanMessage, AIMessage

    prompt_template = ChatPromptTemplate([
        ("system", "あなたは優秀なAIアシスタントです。"),
        MessagesPlaceholder("msgs")
    ])

    prompt_value = prompt_template.invoke({"msgs": [HumanMessage(content="こんにちは!")]})
    print(type(prompt_value)) # <class 'langchain_core.prompt_values.ChatPromptValue'>
    print(prompt_value) 
    # messages=[SystemMessage(content='あなたは優秀なAIアシスタントです。', additional_kwargs={}, response_metadata={}), 
    # HumanMessage(content='こんにちは!', additional_kwargs={}, response_metadata={})]

    # 2つ挿入すれば、PromptValue 内のメッセージは合計3つ
    prompt_value = prompt_template.invoke({"msgs": [HumanMessage(content="2+1=?"), AIMessage(content="3だと思います")]},
                                          config={"run_name": "ChatPromptTemplate with MessagePlaceholder"})
    print(prompt_value) 
    # messages=[SystemMessage(content='あなたは優秀なAIアシスタントです。', additional_kwargs={}, response_metadata={}), 
    # HumanMessage(content='2+1=?', additional_kwargs={}, response_metadata={}), 
    # AIMessage(content='3だと思います', additional_kwargs={}, response_metadata={})]

    # 以下と同じになる
    prompt_template = ChatPromptTemplate([
        ("system", "あなたは優秀なAIアシスタントです。"),
        ("placeholder", "{msgs}") # <-- This is the changed part
    ])

    prompt_value = prompt_template.invoke({"msgs": [HumanMessage(content="2+1=?"), AIMessage(content="3だと思います")]}, 
                                          config={"run_name": "placeholder without MessagePlaceholder"})
    print(prompt_value) 

if __name__ == "__main__":
    main_string_prompt_templates()
    main_chat_prompt_templates()
    main_messages_placeholder()