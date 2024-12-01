def save_compiled_state_graph(graph, png_filename: str = "graph_image.png", xray: bool = True):
    try:
        # 画像を生成
        png_image_data = graph.get_graph(xray=xray).draw_mermaid_png()
        
        # 画像をファイルに保存
        with open(png_filename, "wb") as f:
            f.write(png_image_data)
    except Exception:
        print(f"[save_compiled_state_graph] exception occured.")
        # This requires some extra dependencies and is optional
        pass
