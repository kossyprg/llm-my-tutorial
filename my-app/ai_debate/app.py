from dotenv import load_dotenv
load_dotenv()

import chainlit as cl
from utils import save_compiled_state_graph

from typing import List, Tuple
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=300)
# llm = ChatOllama(
#     model="llama3.2:3b",
#     temperature=0.7,
#     base_url="http://host.docker.internal:11434",
# )

class State(TypedDict):
    messages: List[Tuple[str, str]]
    num_conversation_limit: int
    cnt_turn: int
    theme: str
    position_1: str
    position_2: str

def chatbot(state: State):
    msgs = flip_ai_and_human(state["messages"]) # 対戦相手(AI)を HumanMessage として扱うため
    template = ChatPromptTemplate(msgs)
    chain = template | llm | StrOutputParser()

    theme = state["theme"]
    if (state["cnt_turn"] % 2 == 0):
        position = state["position_1"]
    else:
        position = state["position_2"]
    response = chain.invoke({"theme": theme, "position": position})
    msgs.append(("ai", response))
    return {"messages": msgs, 
            "cnt_turn": state["cnt_turn"] + 1}

def flip_ai_and_human(messages: List[Tuple[str, str]]):
    new_messages = [messages[0]] # SystemMessageを入れる
    for role, msg in messages[-4:]: # 直近の4件だけ入れる
        if role == "system":
            continue
        if role == "human":
            new_messages.append(("ai", msg))
        elif role == "ai":
            new_messages.append(("human", msg))
        else:
            # TODO: tool を使えるようにする
            new_messages.append((role, msg))
    return new_messages

def router(state: State):
    if state["cnt_turn"] >= state["num_conversation_limit"] :
        return "finish"
    return "continue"

prompt = """
あなたは優秀なディベーターです。
以下のお題について対戦相手とディベートを行います。

## お題
{theme}

## あなたの立場
{position}

## ルール
- 自分の意見を述べる際には、論理的かつ具体的な例を交えながら主張を展開してください。   
- 相手の主張が主観的であり論理的でない場合は、反駁してください。
- 主張の論理性、具体性、説得力が評価の基準となります。  
- 明確なデータや根拠を示すことで、説得力を高めることが推奨されます。  
- 発言は100文字程度にまとめるようにしてください。  
- 最終的な勝敗は、議論の内容を基に公平に判断されます。  
"""

@cl.on_chat_start
async def handle_chat_start():
    res = await cl.AskUserMessage(
        content="ディベートのお題を入力してください。\n例：「きのこの山」と「たけのこの里」はどちらの方が優れているか", 
        timeout=300,
    ).send()
    theme = res['output']

    res = await cl.AskUserMessage(
        content="ありがとうございます。続いて両者の主張を明確にします。一方の主張を記述してください。\n例：「きのこの山」の方が優れている",
        timeout=300
    ).send()
    position_1 = res['output']

    res = await cl.AskUserMessage(
        content="もう一方の主張を記述してください。\n例：「たけのこの里」の方が優れている",
        timeout=300
    ).send()
    position_2 = res['output']

    graph_builder = StateGraph(State)
    graph_builder.add_node("ai-1", chatbot)
    graph_builder.add_node("ai-2", chatbot)
    graph_builder.add_edge(START, "ai-1")
    graph_builder.add_conditional_edges(
        "ai-1",
        router,
        {"continue": "ai-2", "finish": END},
    )
    graph_builder.add_conditional_edges(
        "ai-2",
        router,
        {"continue": "ai-1", "finish": END},
    )
    graph = graph_builder.compile()

    # # ASCII表示で確認
    # print(graph.get_graph().draw_ascii()) 

    # # グラフを画像にして確認
    # save_compiled_state_graph(graph, png_filename="img/graph.png")

    inputs = {"messages": [("system", prompt)], 
              "num_conversation_limit": 8,
              "cnt_turn": 0,
              "theme": theme,
              "position_1": position_1,
              "position_2": position_2} 

    await cl.Message(
        content="ありがとうございます。準備が整いましたのでこれよりディベートを開始します。",
    ).send()

    # author によってアイコンを変更している
    # Ref https://docs.chainlit.io/customisation/avatars
    author = "ai-1"
    msg = cl.Message(content="", author=author) 
    
    # streaming については以下を参照
    # Ref https://docs.chainlit.io/advanced-features/streaming
    async for ai_msg_chunk, metadata in graph.astream(
        inputs, 
        config={"run_name": "AI debate app"}, # langsmith の表示を変えるだけ
        stream_mode="messages"
    ):
        if metadata["langgraph_node"] != author:
            author = metadata["langgraph_node"]
            await msg.update()
            msg = cl.Message(content="", author=author)
        await msg.stream_token(ai_msg_chunk.content)
    await msg.update() # 出力中の点滅表示を消すのに必要
