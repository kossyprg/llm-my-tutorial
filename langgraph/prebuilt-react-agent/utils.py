def save_compiled_state_graph(graph, png_filename="graph_image.png"):
    try:
        # 画像を生成
        png_image_data = graph.get_graph(xray=True).draw_mermaid_png()
        
        # 画像をファイルに保存
        with open(png_filename, "wb") as f:
            f.write(png_image_data)
    except Exception:
        print(f"[save_compiled_state_graph] exception occured.")
        # This requires some extra dependencies and is optional
        pass

def print_invoke(response):
    messages = response['messages']
    for msg in messages:
        msg.pretty_print()