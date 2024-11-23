# Ref https://python.langchain.com/docs/how_to/tool_calling/
from dotenv import load_dotenv
load_dotenv()

import numpy as np
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

def main():
    @tool
    def gcd(a: int, b: int) -> int:
        """2つの数の最大公約数を求める"""
        return np.gcd(a, b)
    
    @tool
    def lcm(a: int, b:int) -> int:
        """2つの数の最小公倍数を求める"""
        return np.lcm(a, b)
    
    tools = [gcd, lcm]

    llm = ChatOpenAI(model="gpt-4o-mini")

    # ChatModel の bind_tools() メソッドで tool schemas をモデルに渡す
    # コンセプトガイドの (2)Tool Binding に相当
    # Ref https://python.langchain.com/docs/concepts/tool_calling/#key-concepts
    llm_with_tools = llm.bind_tools(tools)

    query = "9と15の最大公約数は何ですか?"
    response = llm_with_tools.invoke(query)

    # 実際にツールが呼ばれるわけではなく、ツールを呼ぶことを決定したに過ぎない。
    # 引数が入力スキーマに準拠していることも確認している。
    # コンセプトガイドの (3)Tool Calling に相当
    # Ref https://python.langchain.com/docs/concepts/tool_calling/#key-concepts
    print(f"response.content: {response.content}") # "" 空。ツールを実行するわけではない。
    print(f"response.tool_calls: {response.tool_calls}")
    # [{'name': 'gcd', 
    #   'args': {'a': 9, 'b': 15}, 
    #   'id': 'call_7myafWgwuI5EzGhWP11qzHXa', 
    #   'type': 'tool_call'}]

    query = "3と4の最小公倍数は何ですか?"
    response = llm_with_tools.invoke(query)
    print(f"response.content: {response.content}") # ""
    print(f"response.tool_calls: {response.tool_calls}")
    # [{'name': 'lcm', 
    #   'args': {'a': 3, 'b': 4}, 
    #   'id': 'call_g4S0j6lzEjLGjRLcpAnHEq6v', 
    #   'type': 'tool_call'}]

    query = "3と4の最小公倍数と9と15の最大公約数をそれぞれ答えよ"
    response = llm_with_tools.invoke(query)
    print(f"response.content: {response.content}") # ""
    print(f"response.tool_calls: {response.tool_calls}")
    # [{'name': 'lcm', 
    #   'args': {'a': 3, 'b': 4}, 
    #   'id': 'call_CZKo2RlMNULlWTqvhn8Qy9ra', 
    #   'type': 'tool_call'}, 
    #  {'name': 'gcd', 
    #   'args': {'a': 9, 'b': 15}, 
    #   'id': 'call_9ez40fXZma4O1M1mIuLYO6pl', 
    #   'type': 'tool_call'}]

    # ツールが呼び出されない例。tool_calls は空。
    query = "魑魅魍魎ってどういう意味ですか？"
    response = llm_with_tools.invoke(query)
    print(f"response.content: {response.content}") # 「魑魅魍魎（ちみもうりょう）」は、日本の伝説や民間信仰に登場する妖怪や霊的存在を指す言葉です。...
    print(f"response.tool_calls: {response.tool_calls}") # []

if __name__ == "__main__":
    main()
