# Ref https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch/
from dotenv import load_dotenv
load_dotenv()

from icecream import ic
from utils import save_compiled_state_graph, print_invoke

def main():
    # ========= Define graph state =========
    from typing import (
        Annotated,
        Sequence,
        TypedDict,
        Literal
    )
    from langchain_core.messages import BaseMessage
    from langgraph.graph.message import add_messages

    class AgentState(TypedDict):
        """The state of the agent."""

        # add_messages は append の意味と思えばよい。
        # 既存のメッセージを上書きする場合は同じ ID を指定する。
        messages: Annotated[Sequence[BaseMessage], add_messages]


    # ========= Define model and tools =========
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool

    model = ChatOpenAI(model="gpt-4o-mini")

    @tool
    def get_weather(city: Literal["北海道", "東京"]):
        """Use this to get weather information."""
        if city == "北海道":
            return "時折雪が降るでしょう"
        elif city == "東京":
            return "終日晴れるでしょう"
        else:
            raise AssertionError("Unknown city")

    tools = [get_weather]
    model = model.bind_tools(tools)


    # ========= Define nodes and edges =========
    import json
    from langchain_core.messages import ToolMessage, SystemMessage
    from langchain_core.runnables import RunnableConfig

    tools_by_name = {tool.name: tool for tool in tools}


    # Define our tool node
    def tool_node(state: AgentState):
        outputs = []
        # AI Message の tool_calls を調べて、
        # 呼ばれたツールをすべて実行して ToolMessage として与える
        for tool_call in state["messages"][-1].tool_calls:
            tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


    # Define the node that calls the model
    def call_model(
        state: AgentState,
        config: RunnableConfig,
    ):
        # this is similar to customizing the create_react_agent with state_modifier, but is a lot more flexible
        system_prompt = SystemMessage(
            "You are a helpful AI assistant, please respond to the users query to the best of your ability!"
        )
        response = model.invoke([system_prompt] + state["messages"], config)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}


    # Define the conditional edge that determines whether to continue or not
    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # ツールが呼ばれてなかったら END へ
        if not last_message.tool_calls:
            return "end"
        # そうでないなら CONTINUE へ
        else:
            return "continue"
    

    # ========= Define the graph =========
    from langgraph.graph import StateGraph, END

    # Define a new graph
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue, # 分岐する条件を記述した関数
        # 最後にマッピングを渡します。  
        # キーは文字列で、値は他のノードです。  
        # `END` は特別なノードで、グラフの処理が終了することを示します。  
        # この仕組みでは、`should_continue` を呼び出し、その出力がこのマッピング内のキーと一致するかを確認します。  
        # 一致したキーに基づいて、対応するノードが次に呼び出されることになります。
        {
            "continue": "tools",
            "end": END,
        },
    )

    workflow.add_edge("tools", "agent")

    from langgraph.checkpoint.memory import MemorySaver
    memory = MemorySaver()

    # コンパイル時に checkpointer を渡してメモリ機能を追加(tutorialにはない)
    graph = workflow.compile(checkpointer=memory)

    # ASCII表示で確認
    print(graph.get_graph().draw_ascii()) 

    # グラフを画像にして確認
    save_compiled_state_graph(graph, png_filename="img/react_agent_from_scratch_graph.png")


    # ========= 使ってみましょう =========
    config = {"configurable": {"thread_id": "123"}, 
              "run_name": "ReAct agent from scratch"}
    inputs = {"messages": [("user", "札幌の天気は？")]}
    res = graph.invoke(inputs, config)
    print_invoke(res)

    # inputs = {"messages": [("user", "私の先程の質問文を覚えていますか？")]}
    # res = graph.invoke(inputs, config)
    # print_invoke(res)

if __name__ == "__main__":
    main()


