from langchain_huggingface import HuggingFaceEmbeddings

def load_hf_embeddings():
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # 50言語に対応, 384次元
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder="./cache_embed_model",
    )

    return embeddings