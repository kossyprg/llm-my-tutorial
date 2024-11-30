# Ref https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/
from dotenv import load_dotenv
load_dotenv()

from icecream import ic

def main():
    # First we initialize the model we want to use.
    from langchain_openai import ChatOpenAI
    model = ChatOpenAI(model="gpt-4o", temperature=0)

    # このチュートリアルでは事前に定義した地点の天気を返すツールを使います。
    from typing import Literal
    from langchain_core.tools import tool

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

    # Define the graph
    from langgraph.prebuilt import create_react_agent
    graph = create_react_agent(model, tools=tools)

    print(graph.get_graph().draw_ascii()) 

    from utils import save_compiled_state_graph
    save_compiled_state_graph(graph, png_filename="img/create_react_agent_graph.png")

    inputs = {"messages": [("user", "北海道の天気は？")]}
    config={"run_name": "How to use the prebuilt ReAct agent"}
    res = graph.invoke(inputs, config)
    ic(res)

if __name__ == "__main__":
    main()


