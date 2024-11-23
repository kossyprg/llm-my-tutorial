# Ref https://python.langchain.com/docs/how_to/tool_results_pass_to_model/
from dotenv import load_dotenv
load_dotenv()

import numpy as np
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage

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
    ai_msg = llm_with_tools.invoke(query)

    messages = [HumanMessage(query)]
    messages.append(ai_msg)

    # ツール名とツールの辞書
    # ToolMessage を作るのに使う
    tools_dict = {"gcd": gcd, "lcm": lcm}

    import pkg_resources
    from packaging import version
    langchain_core_ver = pkg_resources.get_distribution("langchain-core").version
    print(f"[DEBUG] langchain_core_ver: {langchain_core_ver}")

    # tool_call を使って tool を invoke すると、ToolMessageを受け取れる。
    # この機能は langchain-core==0.2.19 で追加された。
    # それ以前は ToolMessage をこちらで構築する必要がある。
    # Ref https://python.langchain.com/docs/how_to/tool_results_pass_to_model/
    # Ref https://python.langchain.com/api_reference/core/messages/langchain_core.messages.tool.ToolMessage.html
    if version.parse(langchain_core_ver) >= version.parse('0.2.19'):
        for tool_call in ai_msg.tool_calls:
            print(f"[DEBUG] tool_call: {tool_call}")
            # {'name': 'gcd', 
            #  'args': {'a': 9, 'b': 15}, 
            #  'id': 'call_jdf3VF2XJebADD6hOyZJHSdM', 
            #  'type': 'tool_call'}
            selected_tool = tools_dict[tool_call["name"].lower()]
            tool_msg = selected_tool.invoke(tool_call) # ツールを実行
            messages.append(tool_msg)
    else:
        print(f"langchain-core version is earlier than 0.2.19. (langchain-core=={langchain_core_ver})")
        for tool_call in ai_msg.tool_calls:
            selected_tool = tools_dict[tool_call["name"].lower()]
            tool_result = selected_tool.invoke(tool_call['args']) # ツールを実行

            # ToolMessage を構築する
            tool_msg = ToolMessage(
                content=str(tool_result), # '3'
                name=tool_call["name"],   # 'gcd'
                tool_call_id=tool_call["id"] # モデルが生成した id と一致させる必要がある。
            )
            messages.append(tool_msg)
    
    print(f"[DEBUG] messages: {messages}")
    # [HumanMessage(content='9と15の最大公約数は何ですか?', additional_kwargs={}, response_metadata={}), 
    # AIMessage(content='', ...), 
    # ToolMessage(content='3', name='gcd', tool_call_id='call_U1lfTYOFAxyL5vG0fMYWryy1')]

    # tool の出力結果と合わせて最終出力を生成する
    final_output = llm_with_tools.invoke(messages)
    print(final_output.content)
    # 9と15の最大公約数は3です。

if __name__ == "__main__":
    main()
