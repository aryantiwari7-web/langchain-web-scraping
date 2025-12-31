# LangChain と Bright Data を使用した Webスクレイピング

[![Promo](https://github.com/bright-jp/LinkedIn-Scraper/raw/main/Proxies%20and%20scrapers%20GitHub%20bonus%20banner.png)](https://brightdata.jp/) 

このガイドでは、Webスクレイピングを LangChain と組み合わせて、実運用の LLM データエンリッチメントを行う方法を、詳細なステップバイステップで解説します。

- [Webスクレイピングで LLM アプリケーションを強化する](#using-web-scraping-to-power-your-llm-applications)
- [LangChain でスクレイピングデータを使用するメリットと課題](#benefits-and-challenges-of-using-scraped-data-in-langchain)
- [Bright Data による LangChain Webスクレイピング：ステップバイステップガイド](#langchain-web-scraping-powered-by-bright-data-step-by-step-guide)
  - [前提条件](#prerequisites)
  - [ステップ #1: プロジェクトのセットアップ](#step-1-project-setup)
  - [ステップ #2: 必要なライブラリをインストールする](#step-2-install-the-required-libraries)
  - [ステップ #3: プロジェクトを準備する](#step-3-prepare-your-project)
  - [ステップ #4: Web Scraper API を設定する](#step-4-configure-web-scraper-api)
  - [ステップ #5: Webスクレイピングに Bright Data を使用する](#step-5-use-bright-data-for-web-scraping)
  - [ステップ #6: OpenAI モデルを使用する準備をする](#step-6-get-ready-to-use-openai-models)
  - [ステップ #7: LLM プロンプトを生成する](#step-7-generate-the-llm-prompt)
  - [ステップ #8: OpenAI を統合する](#step-8-integrate-openai)
  - [ステップ #9: AI で処理したデータをエクスポートする](#step-9-export-the-ai-processed-data)
  - [ステップ #10: ログを追加する](#step-10-add-logs)
  - [ステップ #11: すべてをまとめる](#step-11-put-it-all-together)
- [結論](#conclusion)


## Webスクレイピングで LLM アプリケーションを強化する

Webスクレイピングは Web サイトからデータを抽出し、RAG（[Retrieval-Augmented Generation](https://brightdata.jp/blog/web-data/rag-explained)）アプリケーションを支え、LLM（[Large Language Models](https://www.ibm.com/think/topics/large-language-models)）を活用するために用いられます。これは、静的なデータベースと、これらのアプリケーションが必要とするリアルタイムでドメイン特化、または大規模なデータセットとのギャップを埋めるものです。

## LangChain でスクレイピングデータを使用するメリットと課題

[LangChain](https://www.langchain.com/) は、分析、要約、Q&A といったタスクのために、LLM を多様なデータソースと統合します。しかし、高品質なデータの収集は、アンチボット対策、CAPTCHA、動的 Web サイトの存在により困難です。Bright Data の [Web Scraper API](https://brightdata.jp/products/web-scraper) は、IP ローテーション、CAPTCHA 解決、JavaScript レンダリングなどの機能でこれらの問題に対処し、シンプルな API 呼び出しを通じて効率的かつ信頼性の高いデータ収集を実現します。

## Bright Data による LangChain Webスクレイピング：ステップバイステップガイド

Bright Data の Web Scraper API を使用して CNN の記事からコンテンツを取得し、その後 OpenAI に送って要約する LangChain の Webスクレイピングスクリプトを構築する方法を学びます。ターゲットとして [この CNN 記事](https://www.cnn.com/2024/12/16/weather/white-christmas-forecast-climate/) を使用します。

![CNN article on Christmas](https://github.com/bright-jp/langchain-web-scraping/blob/main/Images/image-131-1024x492.png)

このシンプルな例は、SERP データに基づく AG チャットボットの作成など、追加の LangChain 機能で簡単に拡張できます。

### 前提条件

このガイドを進めるには、以下が必要です。

- マシンに Python 3+ がインストールされていること
- OpenAI API key
- Bright Data アカウント

### ステップ #1: プロジェクトのセットアップ

Python 3 がインストールされていることを確認してください。次に、プロジェクト用のフォルダを作成します。

```bash
mkdir langchain_scraping
```

`langchain_scrping` に Python の LangChain スクレイピングプロジェクトを格納します。

次に、プロジェクトフォルダへ移動し、その中で Python 仮想環境を初期化します。

```bash
cd langchain_scraping
python3 -m venv env
```

> **Note**:
>
> Windows では、`python3` の代わりに `python` を使用してください。

ここで、お使いの Python IDE でプロジェクトディレクトリを開き、`langchain_scraping` の中に `script.py` ファイルを追加します。

仮想環境を有効化します。

```bash
./env/bin/activate
```

または Windows の場合は、次を実行します。

```bash
env/Scripts/activate
```

### ステップ #2: 必要なライブラリをインストールする

Python の LangChain スクレイピングプロジェクトは、以下のライブラリに依存します。

- [`python-dotenv`](https://pypi.org/project/python-dotenv/): `.env` ファイルから環境変数を読み込むために使用します。Bright Data と OpenAI の認証情報などの機密情報を管理するために使用します。
- [`requests`](https://pypi.org/project/requests/): Bright Data の Web Scraper API とやり取りするための HTTP リクエストを実行します。
- [`langchain_openai`](https://pypi.org/project/langchain-openai/): [`openai`](https://pypi.org/project/openai/) SDK を介した OpenAI 向け LangChain 統合です。

有効化された仮想環境で、依存関係をすべてインストールします。

```bash
pip install python-dotenv requests langchain-community
```

### ステップ #3: プロジェクトを準備する

`scripts.py` に、次の import を追加します。

```python
from dotenv import load_dotenv
import os
```

この 2 行により、環境変数ファイルを読み取れるようになります。

プロジェクトフォルダ内に `.env` ファイルを作成し、すべての認証情報を保存します。

`script.py` で、`python-dotenv` に `.env` から環境変数を読み込むよう指示します。

```python
load_dotenv()
```

これで、`.env` ファイルまたはシステムから次のように環境変数を読み取れます。

```python
os.environ.get("<ENV_NAME>")
```

### ステップ #4: Web Scraper API を設定する

Bright Data の Web Scraper APIs を使用すると、100 以上の Web サイトから解析済みコンテンツを簡単に取得できます。

Web Scraper API をセットアップするには、[公式ドキュメント](https://docs.brightdata.com/scraping-automation/web-data-apis/web-scraper-api/overview) を参照するか、以下の手順に従ってください。

まだ Bright Data アカウントがない場合は作成してください。ログイン後、アカウントのダッシュボードに移動します。ここで左側の「Web Scraper API」ボタンをクリックします。

![Choosing Web Scraper API from the menu on the left](https://github.com/bright-jp/langchain-web-scraping/blob/main/Images/image-133-1024x489.png)

ターゲットサイトは [CNN.com](http://cnn.com/) なので、検索入力に「cnn」と入力し、「CNN news — Collecy by URL」スクレイパーを選択します。

![Searching for hte CNN Scraper API](https://github.com/bright-jp/langchain-web-scraping/blob/main/Images/image-134-1024x486.png)

現在のページで **Create token** ボタンをクリックして [Bright Data API token](https://docs.brightdata.com/general/account/api-token) を生成します。

![Creating a new token for the API](https://github.com/bright-jp/langchain-web-scraping/blob/main/Images/image-135-1024x408.png)

次のモーダルが開き、トークンの詳細を設定できます。

![Configuring the details of the new token](https://github.com/bright-jp/langchain-web-scraping/blob/main/Images/image-136.png)

完了したら **Save** をクリックし、Bright Data API token の値をコピーします。

![Once you clicked save, the new token is shown](https://github.com/bright-jp/langchain-web-scraping/blob/main/Images/image-137.png)

`.env` ファイルに、以下のようにこの情報を保存します。

```python
BRIGHT_DATA_API_TOKEN="<YOUR_BRIGHT_DATA_API_TOKEN>"
```

`<YOUR_BRIGHT_DATA_API_TOKEN>` を、モーダルからコピーした値に置き換えてください。

これで、CNN news の Web Scraper API ページは次の例のようになります。

![The CNN Scraper API page ](https://github.com/bright-jp/langchain-web-scraping/blob/main/Images/image-138-1024x492.png)

### ステップ #5: Webスクレイピングに Bright Data を使用する

Web Scraper API はニーズに合わせたタスクを開始し、その後スクレイピングしたデータのスナップショットを生成します。プロセスの概要は次のとおりです。

1. **リクエスト送信:** スクレイピングするページの URL を指定します。
2. **タスク起動:** API が指定された URL からデータを取得して解析します。
3. **スナップショット取得:** タスク完了後に結果を得るため、スナップショット API を継続的にクエリします。

CNN Web Scraper API の POST エンドポイントは次のとおりです。

```
"https://api.brightdata.com/datasets/v3/trigger?dataset_id=gd_lycz8783197ch4wvwg&include_errors=true"
```

このエンドポイントは `url` フィールドを含むオブジェクトの配列を受け取り、次のようなレスポンスを返します。

```json
{"snapshot_id":"<YOUR_SNAPSHOT_ID>"}
```

このレスポンスの `snapshot_id` を使用して、次のエンドポイントをクエリし、データを取得する必要があります。

```
https://api.brightdata.com/datasets/v3/snapshot/<YOUR_SNAPSHOT_ID>?format=json
```

このエンドポイントは、タスクが進行中の場合は HTTP ステータスコード [`202`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/202) を返し、タスクが完了してデータの準備ができた場合は [`200`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/200) を返します。推奨アプローチは、タスクが終了するまで 10 秒ごとにこのエンドポイントをポーリングすることです。

タスクが完了すると、このエンドポイントは次の形式でデータを返します。

```json
[
    {
        "input": {
            "url": "https://www.cnn.com/2024/12/16/weather/white-christmas-forecast-climate/",
            "keyword": ""
        },
        "id": "https://www.cnn.com/2024/12/16/weather/white-christmas-forecast-climate/index.html",
        "url": "https://www.cnn.com/2024/12/16/weather/white-christmas-forecast-climate/index.html",
        "author": "Mary Gilbert",
        "headline": "White Christmas forecast: Will you be left dreaming of snow or reveling in it?",
        "topics": [
            "weather"
        ],
        "publication_date": "2024-12-16T13:20:52.800Z",
        "updated_last": "2024-12-16T13:20:52.800Z",
        "content": "Christmas is approaching nearly as fast as Santa’s sleigh, but almost anyone in the United States fantasizing about a movie-worthy white Christmas might need to keep dreaming. Early forecasts indicate temperatures could max out around 10 to 15 degrees above normal for much of the country on Christmas Day. [omitted for brevity...]",
        "videos": null,
        "images": [
                "omitted for brevity..."
        ],
        "related_articles": [],
        "keyword": null,
        "timestamp": "2024-12-16T14:18:14.101Z"
    }
]
```

`content` 属性には解析済みの記事データが含まれており、これがアクセスしたい情報です。

これを実装するために、まず `.env` から env を読み取り、エンドポイント URL の定数を初期化します。

```
BRIGHT_DATA_API_TOKEN = os.environ.get("BRIGHT_DATA_API_TOKEN")
BRIGHT_DATA_CNN_WEB_SCRAPER_API_URL = "https://api.brightdata.com/datasets/v3/trigger?dataset_id=gd_lycz8783197ch4wvwg&include_errors=true"
```

次に、上記のプロセスを再利用可能な関数にします。

```python
def get_scraped_data(url):
    # Authorization headers
    headers = {
    "Authorization": f"Bearer {BRIGHT_DATA_API_TOKEN}"
    }
    # Web Scraper API payload
    data = [{
        "url": url
    }]

    # Making the POST request to the Bright Data Web Scraper API
    response = requests.post(BRIGHT_DATA_CNN_WEB_SCRAPER_API_URL, headers=headers, json=data)

    if response.status_code == 200:
        response_data = response.json()
        snapshot_id = response_data.get("snapshot_id")
        if snapshot_id:
            # Iterate until the snapshot is ready
            snapshot_url = f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}?format=json"

            while True:
                snapshot_response = requests.get(snapshot_url, headers=headers)

                if snapshot_response.status_code == 200:
                    # Parse and return the snapshot data
                    snapshot_response_data = snapshot_response.json()
                    return snapshot_response_data[0].get("content")
                elif snapshot_response.status_code == 202:
                    print("Snapshot not ready yet. Retrying in 10 seconds...")
                    time.sleep(10)  # Wait for 10 seconds before retrying
                else:
                    print(f"Failed to retrieve snapshot. Status code: {snapshot_response.status_code}")
                    print(snapshot_response.text)
                    break
        else:
            print("Snapshot ID not found in the response")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
```

動作させるために、次の 2 つの import を追加します。

```python
import requests
import time
```

### ステップ #6: OpenAI モデルを使用する準備をする

この例では、LangChain 内での LLM 統合に OpenAI モデルを使用します。これらのモデルを使用するには、環境変数に OpenAI API key を設定してください。

デフォルトでは、`langchain_openai` は環境変数 [`OPENAI_API_KEY`](https://python.langchain.com/docs/integrations/llms/openai/#credentials) から OpenAI API key を自動的に読み取ります。これをセットアップするには、`.env` ファイルに次の行を追加します。

```
OPENAI_API_KEY="<YOUR_OPEN_API_KEY>"
```

`<YOUR_OPENAI_API_KEY>` を [OpenAI API key](https://platform.openai.com/api-keys) の値に置き換えてください。取得方法が分からない場合は、[公式ガイド](https://platform.openai.com/docs/quickstart) に従ってください。

### ステップ #7: LLM プロンプトを生成する

スクレイピングしたデータを受け取り、記事の要約を取得するためのプロンプトを生成する関数を定義します。

```python
def create_summary_prompt(content, words=100):
    return f"""Summarize the following content in less than {words} words.

           CONTENT:
           '{content}'
           """
```

この例では、完全なプロンプトは次のとおりです。

```
Summarize the following content in less than 100 words.

CONTENT:
'Christmas is approaching nearly as fast as Santa’s sleigh, but almost anyone in the United States fantasizing about a movie-worthy white Christmas might need to keep dreaming. Early forecasts indicate temperatures could max out around 10 to 15 degrees above normal for much of the country on Christmas Day. It’s a forecast reminiscent of last Christmas for many, which came amid the warmest winter on record in the US. But the country could be split in two by warmth and cold in the run up to the big day. [omitted for brevity...]'
```

ChatGPT に渡すと、次のように表示されます。

![Passing the task of summarizing the content in less than 100 words](https://github.com/bright-jp/langchain-web-scraping/blob/main/Images/image-139-1024x626.png)

### ステップ #8: OpenAI を統合する

まず、`get_scraped_data()` 関数を呼び出して、記事ページからコンテンツを取得します。

```python
article_url = "https://www.cnn.com/2024/12/16/weather/white-christmas-forecast-climate/"
scraped_data = get_scraped_data(article_url)
```

`scraped_data` が `None` でなければ、プロンプトを生成します。

```python
if scraped_data is not None:
    prompt = create_summary_prompt(scraped_data)
```

最後に、[GPT-4o mini](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/) AI モデルで設定した [`ChatOpenAI`](https://python.langchain.com/docs/integrations/chat/openai/) の LangChain オブジェクトに渡します。

```python
model = ChatOpenAI(model="gpt-4o-mini")
response = model.invoke(prompt)
```

`langchain_openai` から `ChatOpenAI` を import します。

```python
from langchain_openai import ChatOpenAI
```

処理の最後に、`summary` には前のステップで ChatGPT が生成した要約に近い内容が入ります。

```python
summary = response.content
```

### ステップ #9: AI で処理したデータをエクスポートする

次に、LangChain を介して選択した AI モデルが生成したデータを、JSON ファイルなどの人間が読める形式にエクスポートする必要があります。

そのために、必要なデータを含む辞書を初期化します。次に、以下のようにエクスポートして JSON ファイルとして保存します。

```python
export_data = {
    "url": article_url,
    "summary": summary
}

file_name = "summary.json"
with open(file_name, "w") as file:
    json.dump(export_data, file, indent=4)
```

Python Standard Library から [`json`](https://docs.python.org/3/library/json.html) を import します。

```python
import json
```

### ステップ #10: ログを追加する

Web Scraping AI と ChatGPT 分析を使用したスクレイピングプロセスには時間がかかる場合があります。スクリプトの進行状況を追跡するために、スクリプトの主要ステップで `print()` 文を追加してログを含めます。

```python
article_url = "https://www.cnn.com/2024/12/16/weather/white-christmas-forecast-climate/"
print(f"Scraping data from '{article_url}'...")
scraped_data = get_scraped_data(article_url)

if scraped_data is not None:
    print("Data successfully scraped, creating summary prompt")
    prompt = create_summary_prompt(scraped_data)

    # Ask ChatGPT to perform the task specified in the prompt
    print("Sending prompt to ChatGPT for summarization")
    model = ChatOpenAI(model="gpt-4o-mini")
    response = model.invoke(prompt)

    # Get the AI result
    summary = response.content
    print("Received summary from ChatGPT")

    # Export the produced data to JSON
    export_data = {
        "url": article_url,
        "summary": summary
    }

    print("Exporting data to JSON")
    # Write the output dictionary to JSON file
    file_name = "summary.json"
    with open(file_name, "w") as file:
        json.dump(export_data, file, indent=4)
    print(f"Data exported to '${file_name}'")
else:
    print("Scraping failed")
```

### ステップ #11: すべてをまとめる

最終的な `script.py` ファイルには次の内容が含まれているはずです。

```python
from dotenv import load_dotenv
import os
import requests
import time
from langchain_openai import ChatOpenAI
import json

load_dotenv()

BRIGHT_DATA_API_TOKEN = os.environ.get("BRIGHT_DATA_API_TOKEN")
BRIGHT_DATA_CNN_WEB_SCRAPER_API_URL = "https://api.brightdata.com/datasets/v3/trigger?dataset_id=gd_lycz8783197ch4wvwg&include_errors=true"

def get_scraped_data(url):
    # Authorization headers
    headers = {
        "Authorization": f"Bearer {BRIGHT_DATA_API_TOKEN}"
    }

    # Web Scraper API payload
    data = [{
        "url": url
    }]

    # Making the POST request to the Bright Data Web Scraper API
    response = requests.post(BRIGHT_DATA_CNN_WEB_SCRAPER_API_URL, headers=headers, json=data)

    if response.status_code == 200:
        response_data = response.json()
        snapshot_id = response_data.get("snapshot_id")
        if snapshot_id:
            # Iterate until the snapshot is ready
            snapshot_url = f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}?format=json"

            while True:
                snapshot_response = requests.get(snapshot_url, headers=headers)

                if snapshot_response.status_code == 200:
                    # Parse and return the snapshot data
                    snapshot_response_data = snapshot_response.json()
                    return snapshot_response_data[0].get("content")
                elif snapshot_response.status_code == 202:
                    print("Snapshot not ready yet. Retrying in 10 seconds...")
                    time.sleep(10)  # Wait for 10 seconds before retrying
                else:
                    print(f"Failed to retrieve snapshot. Status code: {snapshot_response.status_code}")
                    print(snapshot_response.text)
                    break
        else:
            print("Snapshot ID not found in the response")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def create_summary_prompt(content, words=100):
    return f"""Summarize the following content in less than {words} words.

           CONTENT:
           '{content}'
           """

# Retrieve the content from the given web page
article_url = "https://www.cnn.com/2024/12/16/weather/white-christmas-forecast-climate/"
scraped_data = get_scraped_data(article_url)

# Ask ChatGPT to perform the task specified in the prompt
prompt = create_summary_prompt(scraped_data)
model = ChatOpenAI(model="gpt-4o-mini")
response = model.invoke(prompt)

# Get the AI result
summary = response.content

# Export the produced data to JSON
export_data = {
    "url": article_url,
    "summary": summary
}

# Write dictionary to JSON file
with open("summary.json", "w") as file:
    json.dump(export_data, file, indent=4)
```

このコマンドで動作確認してください。

```bash
python3 script.py
```

または Windows の場合：

```powershell
python script.py
```

ターミナルの出力は次のものに近いはずです。

```
Scraping data from 'https://www.cnn.com/2024/12/16/weather/white-christmas-forecast-climate/'...
Snapshot not ready yet. Retrying in 10 seconds...
Data successfully scraped, creating summary prompt
Sending prompt to ChatGPT for summarization
Received summary from ChatGPT
Exporting data to JSON
Data exported to 'summary.json'
```

プロジェクトディレクトリに表示された `open.json` ファイルを開くと、次のような内容が表示されるはずです。

```json
{
    "url": "https://www.cnn.com/2024/12/16/weather/white-christmas-forecast-climate/",
    "summary": "As Christmas approaches, forecasts indicate temperatures in the US may be 10 to 15 degrees above normal, continuing a trend from last year\u2019s warm winter. The western US will likely remain warm, while the East experiences colder conditions leading up to Christmas. Some areas may see a mix of rain and snow, but a true \"white Christmas\" requires at least an inch of snow on the ground. Historically, cities like Minneapolis and Burlington have the best chances for snow, while places like New York City and Atlanta have significantly lower probabilities."
}
```

## 結論

このアプローチにはいくつかの課題があります。

- **構造の変更:** Web サイトは頻繁にレイアウトを更新します。
- **アンチボット対策:** 高度な防御が一般的です。
- **スケーラビリティ:** 大量のデータを抽出することは複雑でコストが高くなりがちです。

Bright Data の Web Scraper API はこれらのハードルを克服し、RAG および LangChain を活用したソリューションにとって非常に有用なツールとなります。

登録して、追加の [offerings for AI and LLM](https://brightdata.jp//use-cases/data-for-ai) もぜひご確認ください。