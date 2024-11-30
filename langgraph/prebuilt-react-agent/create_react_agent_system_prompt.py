# Ref https://langchain-ai.github.io/langgraph/how-tos/create-react-agent-system-prompt/
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

    # システムプロンプトを追加します
    prompt = """
これからあなたは『ONE PIECE』の主人公モンキー・D・ルフィのように話すキャラクターとして振る舞います。
以下の指示に従ってください：

- 砕けた口調で話し、難しい言葉は使わない。
- 明るく、陽気で、まっすぐな性格を反映するような回答をする。
- 必要ならば「海賊王になる！」や「仲間」など、ルフィの特徴的なフレーズを使って回答を盛り上げる。
- 一人称は「おれ」、二人称は「お前」または「仲間」とする。
- 余計な敬語は使わず、親しみやすい話し方を意識する。
- 複雑な理屈を省き、簡潔で直感的な回答を心がける。
- ユーザーの質問に対し、ルフィらしい答えをすることを忘れないようにしてください！
"""

    # Define the graph
    from langgraph.prebuilt import create_react_agent
    graph = create_react_agent(model, tools=tools, state_modifier=prompt)

    print(graph.get_graph().draw_ascii()) 

    from utils import save_compiled_state_graph
    save_compiled_state_graph(graph, png_filename="img/create_react_agent_system_prompt_graph.png")

    inputs = {"messages": [("user", "北海道の天気は？")]}
    config = {"run_name": "How to add a custom system prompt to the prebuilt ReAct agent"}
    
    res = graph.invoke(inputs, config)
    ic(res)

if __name__ == "__main__":
    main()


