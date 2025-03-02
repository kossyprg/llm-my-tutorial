## 実行方法

1. `.env` ファイルを作成して環境変数を記述してください。

```
OPENAI_API_KEY="<your-openai-api-key>"

# Langsmithでトレースする場合は以下4つが必要
# LANGCHAIN_PROJECTは任意の名前を設定できる
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="<your-langsmith-api-key>"
LANGCHAIN_PROJECT="chainlit-human-in-the-loop"
```

2. `Dockerfile` を使用してビルドします。

```bash
docker build -t cl-hitl-app .
```

3. ビルドしたイメージを実行してください。`-v`オプションでボリュームをマウントすると、ソースコードの修正がコンテナ環境にも反映されます。

Windows(powershell)の場合
```sh
docker run -it --rm -v "${PWD}:/home/user/app" -p 8000:8000 cl-hitl-app /bin/bash
```

Linuxの場合
```bash
docker run -it --rm -v "$(pwd)":/home/user/app -p 8000:8000 cl-hitl-app /bin/bash
```

4. app.py を実行し、[http://localhost:8000/](http://localhost:8000/) にアクセスしてください。

```bash
chainlit run app.py --host 0.0.0.0
```

5. 終了する際は`exit`を入力してください

```bash
exit
```

