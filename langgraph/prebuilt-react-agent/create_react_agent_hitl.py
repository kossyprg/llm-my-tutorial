# Ref https://langchain-ai.github.io/langgraph/how-tos/create-react-agent-hitl/
from dotenv import load_dotenv
load_dotenv()

from utils import save_compiled_state_graph, print_invoke
from icecream import ic

def main():
    # First we initialize the model we want to use.
    from langchain_openai import ChatOpenAI
    model = ChatOpenAI(model="gpt-4o", temperature=0)

    # このチュートリアルでは事前に定義した地点の天気を返すツールを使います。
    from typing import Literal
    from langchain_core.tools import tool

    @tool
    def get_weather(city: str):
        """Use this to get weather information."""
        if city == "北海道":
            return "時折雪が降るでしょう"
        elif city == "東京":
            return "終日晴れるでしょう"
        else:
            raise AssertionError("Unknown city")

    ic(get_weather.args) # {'city': {'title': 'City', 'type': 'string'}}

    tools = [get_weather]

    # human-in-the-loop(HITL) の実装するためには checkpointer が必要
    from langgraph.checkpoint.memory import MemorySaver
    memory = MemorySaver()

    # Define the graph
    from langgraph.prebuilt import create_react_agent
    graph = create_react_agent(
        model, tools=tools, interrupt_before=["tools"], checkpointer=memory
    )

    print(graph.get_graph().draw_ascii()) 
    save_compiled_state_graph(graph, png_filename="img/create_react_agent_hitl_graph.png")

    inputs = {"messages": [("user", "札幌の天気は？")]}
    config = {"configurable": {"thread_id": "1"}, 
              "run_name": "How to add HITL processes to the prebuilt ReAct agent"}
    
    res = graph.invoke(inputs, config)
    ic(res['messages'][-1].tool_calls)
    # [{'args': {'city': '札幌'},
    #   'id': 'call_KyFxUuUjbvEokCrQsSws0zj1',
    #   'name': 'get_weather',
    #   'type': 'tool_call'}]

    snapshot = graph.get_state(config)
    ic("Next step: ", snapshot.next) # ('tools',)

    last_message = snapshot.values["messages"][-1]
    last_message.tool_calls[0]["args"] = {"city": "北海道"} # 北海道にすり替える
    graph.update_state(config, {"messages": [last_message]})

    # 続行
    res = graph.invoke(None, config)
    print_invoke(res)

if __name__ == "__main__":
    main()


