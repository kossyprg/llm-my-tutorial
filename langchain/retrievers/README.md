## Retrievers チュートリアル

Retrievers に関するチュートリアルを実行するためのソースファイル群です。

参考：[langchain Retrievers](https://python.langchain.com/docs/how_to/#retrievers)

## 実行方法

1. `.env` ファイルを作成して環境変数を記述してください。

```
OPENAI_API_KEY="<your-openai-api-key>"
HF_TOKEN="<your-hf-token>"

# Langsmithでトレースする場合は以下4つが必要
# LANGCHAIN_PROJECTは任意の名前を設定できる
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="<your-langsmith-api-key>"
LANGCHAIN_PROJECT="retrievers-tutorial"
```

2. `Dockerfile` を使用してビルドします。

```bash
docker build -t retrievers .
```

3. ビルドしたイメージを実行してください。`-v`オプションでボリュームをマウントすると、ソースコードの修正がコンテナ環境にも反映されます。

Windows(cmd)の場合
```cmd
docker run -it --rm -v "%cd%":/home/user/app retrievers /bin/bash
```

4. 所望のスクリプトを実行してください。

```bash
python custom_retriever.py
```

5. 終了する際は`exit`を入力してください

```bash
exit
```

## ソースコード

### Retriever を自作する
[custom_retriever.py](custom_retriever.py)

参考：[How to create a custom Retriever](https://python.langchain.com/docs/how_to/custom_retriever/)

`BaseRetriever` を継承して `Retriever` を自作する方法です。
`_get_relevant_documents()` は必須です。

```python
from langchain_core.retrievers import BaseRetriever

class ToyRetriever(BaseRetriever):
    documents: List[Document]
    """List of documents to retrieve from."""
    k: int
    """Number of top results to return"""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # 種々の処理
        # メンバ変数は self.k のようにアクセスする
        return matching_documents
```
### 類似度スコアを付与する方法
[add_scores_retriever.py](add_scores_retriever.py)

参考：[How to add scores to retriever results](https://python.langchain.com/docs/how_to/add_scores_retriever/)

`Document` のメタデータに類似度スコアが付与されるようにする方法です。
`similarity_search_with_score()` を使うことで対応できます。

戻り値は `List[Tuple[Document, float]]` なので、`zip()` などを使ってメタデータとして格納します。

```python
@chain
def retriever(query: str) -> List[Document]:
    # zipについてはzip_example()を参照
    # docs: Tuple[Document], scores: Tuple[Float]
    docs, scores = zip(*vectorstore.similarity_search_with_score(query)) 

    # docのmetadataにscoreキーを追加して、そこに類似度スコアを格納
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score

    return docs
```

### 小さいチャンクで検索をかけて大きいチャンクを取得する方法

[parent_document_retriever.py](parent_document_retriever.py)

参考：[How to use the Parent Document Retriever](https://python.langchain.com/docs/how_to/parent_document_retriever/)

ベクトル検索を行う際には、以下のような相反する2つの要件を満たす必要があります。

1. ベクトル検索の精度を確保：各チャンクの意味が失われないように、適度に短いチャンクで検索を行いたい。
2. LLMの文脈理解を確保：文脈が失われないように、適度に長いチャンクをLLMに渡したい。

この両者を実現する方法の一つに `ParentDocumentRetriever` があります。この `Retriever` は子チャンク（小さいチャンク）でベクトル検索をかけ、それに対応する親チャンク（大きいチャンク）を返します。

親チャンクと子チャンクそれぞれの `TextSplitter` を定義し、`ParentDocumentRetriever` に渡します。親チャンクは `InMemoryStore()` に格納され、ヒットした子チャンクの `doc_id` を用いて親チャンクを取得します。

```python
from langchain.retrievers import ParentDocumentRetriever

loader = TextLoader("./run_melos.txt", encoding="utf-8")
documents = loader.load()

parent_splitter = RecursiveCharacterTextSplitter(
    separators=["。"],
    chunk_size=500,
    chunk_overlap=0,
)

child_splitter = RecursiveCharacterTextSplitter(
    separators=["。"], 
    chunk_size=15,   # 簡単のため、極端に小さくしている
    chunk_overlap=0,
)

store = InMemoryStore()

# k は子チャンクの検索個数を表し、k = 子チャンクの検索個数 >= 親チャンクの検索個数である。
# 子チャンクの検索個数 > 親チャンクの検索個数 となるのは、
# ヒットした子チャンクの属する親チャンクが重複した場合。
k = 6
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    parent_splitter=parent_splitter,
    child_splitter=child_splitter,
    search_kwargs={"k": k}
)

retriever.add_documents(documents=documents)
query = "セリヌンティウス"
```

`ParentDocumentRetriever` の準備が整いました。検証目的で `retriever.vectorstore` でベクトル検索をかけてみます。

```python
# （デバッグ用）ドキュメントをコンソールに表示する
def print_docs(docs, label="chunk"):
    for i, doc in enumerate(docs):
        print("="*5 + f"{label} {i+1}" + "="*5)
        print(doc.page_content)
        print(doc.metadata)

# （デバッグ用）子チャンクの検索結果
sub_docs = retriever.vectorstore.similarity_search(query, k = k)
print_docs(sub_docs, label="child chunk")
```

`ParentDocumentRetriever` の持つ `vectorstore` には**子チャンク**の埋め込みが登録されていることに注意します。
例えば、ヒットした子チャンクは以下のようになります。`doc_id` は親チャンクと対応付けるための UUID です。

```bash
=====child chunk 1=====
。「セリヌンティウス
{'doc_id': '6c9ef208-c5c6-4c76-a01c-f92dffe4aaa9', 'source': './run_melos.txt'}
=====child chunk 2=====
。セリヌンティウスである
{'doc_id': '9b93ae5e-1692-4a2c-b234-19bd0206e9d3', 'source': './run_melos.txt'}
=====child chunk 3=====
。セリヌンティウスよ、ゆるしてくれ
{'doc_id': '1a9d3228-868f-49d4-8472-88606405c0f8', 'source': './run_melos.txt'}
=====child chunk 4=====
。やんぬる哉(かな)
{'doc_id': '5e888cac-f418-44e0-b26d-2a2f068a22af', 'source': './run_melos.txt'}
=====child chunk 5=====
。セリヌンティウスの縄は、ほどかれたのである
{'doc_id': '6c9ef208-c5c6-4c76-a01c-f92dffe4aaa9', 'source': './run_melos.txt'}
=====child chunk 6=====
。）
{'doc_id': '3fd52eb2-6bf4-4cfe-b270-edfd596c7cd6', 'source': './run_melos.txt'}
```

`ParentDocumentRetriever` の `invoke()` を実行すると、親チャンクが得られます。

```python
# 子チャンクでベクトル検索をかけて、
# ヒットした子チャンクの属する親チャンクを取得する
retrieved_docs = retriever.invoke(query)
print_docs(retrieved_docs, label="parent chunk")
```

`k=6` を指定しましたが、最終的に取得した親チャンクは5個でした。これは先程得られた子チャンクに重複があるためです（`child chunk 1` と `child chunk 5` の `doc_id` が同じ）。
```bash
=====parent chunk 1=====
。」と大声で刑場の群衆にむかって叫んだつもりであったが、喉(のど)がつぶれて嗄(しわが)れた声が幽(かす)かに出たばかり、群衆は、ひとりとして彼の到着に気がつかない。すでに磔の柱が高々と立てられ、縄を打たれたセリヌンティウスは、徐々に釣り上げられてゆく。メロスはそれを目撃して最後の勇、先刻、濁流を泳いだように群衆を掻きわけ、掻きわけ、「私だ、刑吏！　殺されるのは、私だ。メロスだ。彼を人質にした私は、ここにいる！」と、かすれた声で精一ぱいに叫びながら、ついに磔台に昇り、釣り上げられてゆく友の両足に、齧(かじ)りついた。群衆は、どよめいた。あっぱれ。ゆるせ、と口々にわめいた。セリヌンティウスの縄は、ほどかれたのである。「セリヌンティウス。」メロスは眼に涙を浮べて言った。「私を殴れ。ちから一ぱいに頬を殴れ。私は、途中で一度、悪い夢を見た。君が若(も)し私を殴ってくれなかったら、私は君と抱擁する資格さえ無いのだ。殴れ。」セリヌンティウスは、すべてを察した様子で首肯(うなず)き、刑場一ぱいに鳴り響くほど音高くメロスの右頬を殴った。殴ってから優しく微笑(ほほえ)み、「メロス、私を殴れ
{'source': './run_melos.txt'}
=====parent chunk 2=====
メロスは激怒した。必ず、かの邪智暴虐(じゃちぼうぎゃく)の王を除かなければならぬと決意した。メロスには政治がわからぬ。メロスは、村の牧人である。笛を吹き、羊と遊んで暮して来た。けれども邪悪に対しては、人一倍に敏感であった。きょう未明メロスは村を出発し、野を越え山越え、十里はなれた此(こ)のシラクスの市にやって来た。メロスには父も、母も無い。女房も無い。十六の、内気な妹と二人暮しだ。この妹は、村の或る律気な一牧人を、近々、花婿(はなむこ)として迎える事になっていた。結婚式も間近かなのである。メロスは、それゆえ、花嫁の衣裳やら祝宴の御馳走やらを買いに、はるばる市にやって来たのだ。先ず、その品々を買い集め、それから都の大路をぶらぶら歩いた。メロスには竹馬の友があった。セリヌンティウスである。今は此のシラクスの市で、石工をしている。その友を、これから訪ねてみるつもりなのだ。久しく逢わなかったのだから、訪ねて行くのが楽しみである。歩いているうちにメロスは、まちの様子を怪しく思った。ひっそりしている
{'source': './run_melos.txt'}
=====parent chunk 3=====
。もう、どうでもいいという、勇者に不似合いな不貞腐(ふてくさ)れた根性が、心の隅に巣喰った。私は、これほど努力したのだ。約束を破る心は、みじんも無かった。神も照覧、私は精一ぱいに努めて来たのだ。動けなくなるまで走って来たのだ。私は不信の徒では無い。ああ、できる事なら私の胸を截(た)ち割って、真紅の心臓をお目に掛けたい。愛と信実の血液だけで動いているこの心臓を見せてやりたい。けれども私は、この大事な時に、精も根も尽きたのだ。私は、よくよく不幸な男だ。私は、きっと笑われる。私の一家も笑われる。私は友を欺(あざむ)いた。中途で倒れるのは、はじめから何もしないのと同じ事だ。ああ、もう、どうでもいい。これが、私の定った運命なのかも知れない。セリヌンティウスよ、ゆるしてくれ。君は、いつでも私を信じた。私も君を、欺かなかった。私たちは、本当に佳い友と友であったのだ。いちどだって、暗い疑惑の雲を、お互い胸に宿したことは無かった。いまだって、君は私を無心に待っているだろう。ああ、待っているだろう。ありがとう、セリヌンティウス。よくも私を信じてくれた。それを思えば、たまらない
{'source': './run_melos.txt'}
=====parent chunk 4=====
。妹夫婦は、まさか私を村から追い出すような事はしないだろう。正義だの、信実だの、愛だの、考えてみれば、くだらない。人を殺して自分が生きる。それが人間世界の定法ではなかったか。ああ、何もかも、ばかばかしい。私は、醜い裏切り者だ。どうとも、勝手にするがよい。やんぬる哉(かな)。――四肢を投げ出して、うとうと、まどろんでしまった。ふと耳に、潺々(せんせん)、水の流れる音が聞えた。そっと頭をもたげ、息を呑んで耳をすました。すぐ足もとで、水が流れているらしい。よろよろ起き上って、見ると、岩の裂目から滾々(こんこん)と、何か小さく囁(ささや)きながら清水が湧き出ているのである。その泉に吸い込まれるようにメロスは身をかがめた。水を両手で掬(すく)って、一くち飲んだ。ほうと長い溜息が出て、夢から覚めたような気がした。歩ける。行こう。肉体の疲労恢復(かいふく)と共に、わずかながら希望が生れた。義務遂行の希望である。わが身を殺して、名誉を守る希望である。斜陽は赤い光を、樹々の葉に投じ、葉も枝も燃えるばかりに輝いている。日没までには、まだ間がある。私を、待っている人があるのだ
{'source': './run_melos.txt'}
=====parent chunk 5=====
。この可愛い娘さんは、メロスの裸体を、皆に見られるのが、たまらなく口惜しいのだ。」勇者は、ひどく赤面した。（古伝説と、シルレルの詩から。）
{'source': './run_melos.txt'}
```