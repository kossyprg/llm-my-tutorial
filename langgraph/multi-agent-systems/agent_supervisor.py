# Ref https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/
from dotenv import load_dotenv
load_dotenv()

from icecream import ic
from langchain_core.tools import tool
from typing import Literal
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

from utils import save_compiled_state_graph

def main():
    @tool
    def get_temperature(location: str) -> float:
        """地点の温度を返します"""
        if location == "東京":
            return 33.3
        elif location == "北海道":
            return 11.1
        return -999999
    
    @tool
    def calc_mean(a: float, b: float) -> float:
        """Calculate the arithmetic mean of two floating-point numbers."""
        return (a + b) / 2.0
    
    members = ["weather_reporter", "calculator"]
    # Our team supervisor is an LLM node. It just picks the next agent to process
    # and decides when the work is completed
    options = members + ["FINISH"]

    system_prompt = (
        "あなたは、次の作業者間の会話を管理する役割を持つ監督者です："
        f"{members}。以下のユーザーリクエストに基づき、次に作業を行う"
        "作業者を指定してください。各作業者はタスクを実行し、その結果と"
        "ステータスを返します。作業がすべて完了したら、FINISHと応答してください。"
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]


    llm = ChatOpenAI(model="gpt-4o-mini")


    def supervisor_node(state: MessagesState) -> Command[Literal[*members, "__end__"]]:
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            goto = END

        return Command(goto=goto)

    weather_report_agent = create_react_agent(
        llm,
        tools=[get_temperature],
        state_modifier="あなたは天気予報士です。いかなる算術的演算も行ってはいけません。"
    )


    def weather_report_node(state: MessagesState) -> Command[Literal["supervisor"]]:
        result = weather_report_agent.invoke(state)
        return Command(
            update={
                "messages": [
                    HumanMessage(content=result["messages"][-1].content, name="weather_reporter")
                ]
            },
            goto="supervisor",
        )

    calculator_agent = create_react_agent(llm, tools=[calc_mean])


    def calc_node(state: MessagesState) -> Command[Literal["supervisor"]]:
        result = calculator_agent.invoke(state)
        return Command(
            update={
                "messages": [
                    HumanMessage(content=result["messages"][-1].content, name="calculator")
                ]
            },
            goto="supervisor",
        )


    builder = StateGraph(MessagesState)
    builder.add_edge(START, "supervisor")
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("weather_reporter", weather_report_node)
    builder.add_node("calculator", calc_node)
    graph = builder.compile()

    print(graph.get_graph().draw_ascii())
    save_compiled_state_graph(graph, png_filename="img/agent_supervisor-graph.png", xray=False)
    save_compiled_state_graph(graph, png_filename="img/agent_supervisor-graph-detailed.png", xray=True)

    # サブグラフを含むグラフの実行
    config = {"run_name": "agent supervisor graph"}
    inputs = {"messages": [("user", "東京と北海道の気温を調べてその平均値を求めなさい。")]}
    for chunk in graph.stream(inputs, config=config, subgraphs=True):
        ic(chunk)
    

if __name__ == "__main__":
    main()