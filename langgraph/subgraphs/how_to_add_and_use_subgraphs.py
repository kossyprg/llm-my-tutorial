# Ref https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/
from dotenv import load_dotenv
load_dotenv()
from icecream import ic
from utils import save_compiled_state_graph

def main():
    from langgraph.graph import START, StateGraph, END
    from typing import Annotated, TypedDict, List
    from operator import add

    # ========================== サブグラフ ==========================
    class SubgraphState(TypedDict):
        balance: int
        bag: Annotated[List[str], add]
        coupon: int
    
    def flower_shop(state: SubgraphState):
        balance = state["balance"] - 200
        return {"balance": balance, "bag": ["carnation"], "coupon": 1}

    def supermarket(state: SubgraphState):
        price = 3000
        coupon = state["coupon"]
        if coupon > 0:
            price -= 300
            coupon -= 1
        balance = state["balance"] - price
        stuffs = ["beef", "milk", "potatoes"]
        return {"balance": balance, "bag": stuffs, "coupon": coupon}
    
    subgraph_builder = StateGraph(SubgraphState)
    subgraph_builder.add_node("flower_shop", flower_shop)
    subgraph_builder.add_node("supermarket", supermarket)
    subgraph_builder.add_edge(START, "flower_shop")
    subgraph_builder.add_edge("flower_shop", "supermarket")
    subgraph_builder.add_edge("supermarket", END)
    subgraph = subgraph_builder.compile()

    print(subgraph.get_graph(xray=1).draw_ascii())
    save_compiled_state_graph(subgraph, png_filename="img/how-to-add-and-use-subgraph-subgraph.png")

    config = {"run_name": "Subgraph only"}
    for chunk in subgraph.stream({"balance": 5000}, config=config):
        ic(chunk)

    res = subgraph.invoke({"balance": 5000}, config=config)
    ic(res)

    # ========================== 親グラフ ==========================
    class ParentState(TypedDict):
        balance: int
        bag: Annotated[List[str], add]

    def atm(state: ParentState):
        return {"balance": 10000}

    builder = StateGraph(ParentState)
    builder.add_node("ATM", atm)

    # コンパイルしたサブグラフを、親グラフのノードとして add_node する
    builder.add_node("SHOPS", subgraph)

    builder.add_edge(START, "ATM")
    builder.add_edge("ATM", "SHOPS")
    builder.add_edge("SHOPS", END)
    graph = builder.compile()

    print(graph.get_graph(xray=True).draw_ascii())
    save_compiled_state_graph(graph, png_filename="img/how-to-add-and-use-subgraph-graph.png")

    # サブグラフを含むグラフの実行
    config = {"run_name": "Graph including subgraph"}
    for chunk in graph.stream({"balance": 0}, config=config):
        ic(chunk)

    # サブグラフからの出力も確認する
    for chunk in graph.stream({"balance": 0}, config=config, subgraphs=True):
        ic(chunk)

    res = graph.invoke({"balance": 0}, config=config)
    ic(res)

# 親グラフと全く別のキーを持たせる場合
def main_add_node_function():
    from langgraph.graph import START, StateGraph, END
    from typing import Annotated, TypedDict, List
    from operator import add

    # ========================== サブグラフ ==========================
    class SubgraphState(TypedDict):
        expense: int # 残高ではなく、かかった費用にする
        bag: Annotated[List[str], add]
        coupon: int
    
    def flower_shop(state: SubgraphState):
        expense = state["expense"] + 200
        return {"expense": expense, "bag": ["carnation"], "coupon": 1}

    def supermarket(state: SubgraphState):
        price = 3000
        coupon = state["coupon"]
        if coupon > 0:
            price -= 300
            coupon -= 1
        expense = state["expense"] + price
        stuffs = ["beef", "milk", "potatoes"]
        return {"expense": expense, "bag": stuffs, "coupon": coupon}
    
    subgraph_builder = StateGraph(SubgraphState)
    subgraph_builder.add_node("flower_shop", flower_shop)
    subgraph_builder.add_node("supermarket", supermarket)
    subgraph_builder.add_edge(START, "flower_shop")
    subgraph_builder.add_edge("flower_shop", "supermarket")
    subgraph_builder.add_edge("supermarket", END)
    subgraph = subgraph_builder.compile()

    print(subgraph.get_graph(xray=1).draw_ascii())
    save_compiled_state_graph(subgraph, png_filename="img/how-to-add-and-use-subgraph-no-shared-keys-subgraph.png")

    config = {"run_name": "Subgraph only (no shared keys)"}
    for chunk in subgraph.stream({"expense": 0}, config=config):
        ic(chunk)

    res = subgraph.invoke({"expense": 0}, config=config)
    ic(res)

    # ========================== 親グラフ ==========================
    class ParentState(TypedDict):
        balance: int

    def atm(state: ParentState):
        return {"balance": 10000}

    def shopping(state: ParentState):
        # トレースで確認するため。必須でない
        config = {"run_name": "shopping"}

        response = subgraph.invoke({"expense": 0}, config=config)
        return {"balance": state["balance"] - response["expense"]}
    
    builder = StateGraph(ParentState)
    builder.add_node("ATM", atm)

    # サブグラフを実行する関数を与える
    builder.add_node("SHOPS", shopping)

    builder.add_edge(START, "ATM")
    builder.add_edge("ATM", "SHOPS")
    builder.add_edge("SHOPS", END)
    graph = builder.compile()

    print(graph.get_graph(xray=True).draw_ascii())
    save_compiled_state_graph(graph, png_filename="img/how-to-add-and-use-subgraph-no-shared-keys-graph.png")

    # サブグラフを含むグラフの実行
    config = {"run_name": "Graph including subgraph (no shared keys)"}
    for chunk in graph.stream({"balance": 0}, config=config):
        ic(chunk)

    # サブグラフからの出力も確認する
    for chunk in graph.stream({"balance": 0}, config=config, subgraphs=True):
        ic(chunk)

    res = graph.invoke({"balance": 0}, config=config)
    ic(res)

if __name__ == "__main__":
    # main()

    # 親グラフと全く別のキーを持たせる場合
    main_add_node_function()