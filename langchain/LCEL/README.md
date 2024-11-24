## LCEL チュートリアル

LangChain Expression Language (LCEL) に関するチュートリアルを実行するためのソースファイル群です。

参考：[LangChain Expression Language (LCEL)](https://python.langchain.com/docs/how_to/#langchain-expression-language-lcel)

## 実行方法

1. `.env` ファイルを作成して環境変数を記述してください。

```
OPENAI_API_KEY="<your-openai-api-key>"
HF_TOKEN="<your-hf-token>"

LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="<your-langsmith-api-key>"
LANGCHAIN_PROJECT="LCEL-tutorial"
```

2. `Dockerfile` を使用してビルドします。

```bash
docker build -t lcel .
```

3. ビルドしたイメージを実行してください。`-v`オプションでボリュームをマウントすると、ソースコードの修正がコンテナ環境にも反映されます。

Windows(cmd)の場合
```bash
docker run -it --rm -v "%cd%":/home/user/app --name lcel lcel /bin/bash
```

4. 所望のスクリプトを実行してください。

```bash
python chain_runnables.py
```

5. 終了する際は`exit`を入力してください

```bash
exit
```

## ソースコード

### Runnable を連結する方法

[chain_runnables.py](chain_runnables.py)

参考：[How to chain runnables](https://python.langchain.com/docs/how_to/sequence/)

`Runnable` はパイプ演算子 `|` あるいは `.pipe()` メソッドで連結できます。

```python
model = ChatOpenAI(model="gpt-4o-mini")

system_prompt = "あなたはしりとりゲームを行うAIアシスタントです。" + \
                "最初の単語の最後の文字を次の単語の最初に使います。" + \
                "単語は日本語で、「ん」で終わる単語を言った場合、負けになります。" + \
                "回答はひらがなのみ許可され、漢字を使用してはいけません。" + \
                "また、単語以外の回答やルール違反をしないでください。"

prompt_template = ChatPromptTemplate([
    ("system", system_prompt),
    ("user", "{input}")
])

# パイプ演算子を使って2つ以上の Runnable オブジェクトを連結する
chain = prompt_template | model | StrOutputParser()
# chain = prompt_template.pipe(model).pipe(StrOutputParser()) # 同じ
response = chain.invoke({"input": "りんご"}, config={"run_name": "simple chain"})
print(response) # ごま
```

