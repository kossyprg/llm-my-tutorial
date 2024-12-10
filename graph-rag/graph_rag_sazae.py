# Ref https://qiita.com/ksonoda/items/98a6607f31d0bbb237ef
from dotenv import load_dotenv
load_dotenv()

from icecream import ic

from pydantic import BaseModel, Field
from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Neo4jVector
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

def main():
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini") 
    graph = Neo4jGraph()

    # (必要なら)既にあるグラフを削除する
    delete_before_register = True
    if delete_before_register:
        # Ref https://qiita.com/euonymus/items/82260dbbda4df7c2f68f
        delete_query = "MATCH (n) DETACH DELETE n"
        graph.query(delete_query)
        print("Delete complete")

    # グラフ構造抽出
    add_graph_documents = True
    if add_graph_documents:
        # 適当にチャンク分割
        raw_documents = TextLoader('./sazaesan.txt', encoding="utf-8").load()
        splitter = RecursiveCharacterTextSplitter(separators=["\n\n"], chunk_size=100, chunk_overlap=0)
        documents = splitter.split_documents(raw_documents)

        config = {"run_name": "convert to graph documents"}
        llm_transformer = LLMGraphTransformer(llm=llm)
        graph_documents = llm_transformer.convert_to_graph_documents(documents, config=config)

        # 登録
        graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True
        )

        # インデックスを作成
        graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

    vector_index = Neo4jVector.from_existing_graph(
        OpenAIEmbeddings(model="text-embedding-ada-002"),
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )

    # Extract entities from text
    class Entities(BaseModel):
        """エンティティに関する情報の識別"""

        names: List[str] = Field(
            description="文章の中に登場する、人物、各人物の性格、各人物間の続柄、各人物が所属する組織、各人物の家族関係",
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "テキストから家族と人物のエンティティを抽出します。",
            ),
            (
                "human",
                "指定された形式を使用して、以下から情報を抽出します。"
                "input: {question}",
            ),
        ]
    )

    entity_chain = prompt | llm.with_structured_output(Entities)
    config = {"run_name": "Who are Katsuo's parents?"}
    res = entity_chain.invoke({"question": "カツオの両親は誰ですか？"}, config=config)
    ic(res.names)

    def generate_full_text_query(input: str) -> str:
        """
        指定された入力文字列に対する全文検索クエリを生成します。

        この関数は、全文検索に適したクエリ文字列を構築します。
        入力文字列を単語に分割し、
        各単語に対する類似性のしきい値 (変更された最大 2 文字) を結合します。
        AND 演算子を使用してそれらを演算します。ユーザーの質問からエンティティをマッピングするのに役立ちます
        データベースの値と一致しており、多少のスペルミスは許容されます。
        """
        full_text_query = ""
        words = [el for el in remove_lucene_chars(input).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()

    def structured_retriever(question: str, config=None) -> str:
        """
        質問の中で言及されたエンティティの近傍を収集します。
        """
        result = ""
        entities = entity_chain.invoke({"question": question}, config=config)
        for entity in entities.names:
            response = graph.query(
                """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node,score
                CALL {
                WITH node
                MATCH (node)-[r:!MENTIONS]->(neighbor)
                RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                UNION ALL
                WITH node
                MATCH (node)<-[r:!MENTIONS]-(neighbor)
                RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                }
                RETURN output LIMIT 50
                """,
                {"query": generate_full_text_query(entity)},
            )
            result += "\n".join([el['output'] for el in response])
        return result

    config = {"run_name": "Who are Katsuo's parents? with structured_retriever"}
    res = structured_retriever("カツオの両親は誰ですか？", config=config)
    ic(res)
    
    def retriever(question: str):
        print(f"Search query: {question}")
        structured_data = structured_retriever(question)
        unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
        final_data = f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ". join(unstructured_data)}
"""
        return final_data

    template = """
あなたは優秀なAIです。下記のコンテキストを利用してユーザーの質問に丁寧に答えてください。
必ず文脈からわかる情報のみを使用して回答を生成してください。
{context}

ユーザーの質問: {question}
"""
    prompt = ChatPromptTemplate.from_messages([("human", template)])

    # Graph RAG のパイプライン
    graph_rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    config = {"run_name": "Sazaesan Graph-RAG"}
    res = graph_rag_chain.invoke("マスオが勤めている会社は？", config=config)
    ic(res)

if __name__ == "__main__":
    main()