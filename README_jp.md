# LLM 20 Questions - yukky_maru

このリポジトリには、LLM 20 Questions コンペティションのエージェントとそれを動かすためのコード及びファイルが含まれています。ただし、エージェントで使用するLLMの重みはgithubにアップロードできないため、手動でダウンロードし、更にcongif.jsonの変更が必要です。そのため、私がDockerhubにアップロードしたDockerイメージを使用することをお勧めします。

## 動画で解説

docker imageのプルから実際にllm 20 questionを私のエージェントでテストするまでの動画を作成しました。参考にしてください。
https://youtu.be/D77itV-L44s

## 必要なもの

このリポジトリのコードを使用するにはkaggleの公式環境である`gcr.io/kaggle-gpu-images/python`と追加のpythonライブラリ及びエージェントで使用するLLMの重みが必要です。必要なものは以下の通りです。

* kaggle環境
* **追加のPython ライブラリ:**
    * torch==2.1.2
    * transformers==4.42.3
    * bitsandbytes==0.43.3
    * accelerate==0.33.0
* LLMの重み及び設定ファイル(ただし、一部の設定ファイルは変更が必要です)

### Dockerhubを使用する場合

1. `docker pull yukimaru77/llm-20-questions_5th:latest`

上記のコマンドを用いることで、私がDockerhubにアップロードしたDockerイメージをダウンロードすることができます。

### Dockerhubを使用しない場合

1. このリポジトリをクローンしてください。
2. main.pyと同じディレクトリにweights_VAGOsolutionsというディレクトリを作成し、そこに[こちら](https://huggingface.co/VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct)からLLMの重みを格納してください。
3. weights_VAGOsolutions内のconfig.jsonの"rope_scalingを以下のように変更してください。
```json
  "rope_scaling": {
    "factor": 8.0,
    "type": "dynamic"
  }
```
4. Dockerfileをビルドしてください。
```bash
docker build -t llm-20-questions_5th -f Dockerfile .
```

## 環境構築

1. Docker コンテナを実行します。
```bash
docker run -itd --gpus all --name llm-20-questions_5th -p 8889:8888 yukimaru77/llm-20-questions_5th:latest jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
```

## ハードウェア

私の環境

* **CPU:** AMD EPYC 7702P 64-Core Processor
* **メモリ:** 503 GB
* **GPU:** 4x Quadro RTX 8000

## OS

* **Ubuntu 20.04.6 LTS (Focal Fossa)**

## 予測

ゲームを実行するために必要なagent_fn、及びキーワードリスト、モデルの重みはそれぞれ以下に格納されています。
* agent_fn: `main.py`
* キーワードリスト: `english_nouns_num.pkl、english_nouns.pkl、large_words_frequency.pickle、large_words.pickle`
* モデルの重み: `weights_VAGOsolutions`

## テスト

`test_game.ipynb` を使用して私のエージェントをテストすることが出来ます。/app配下に必要なコードが格納されています。実際にテストをしたい場合はtest_game.ipynbを実行してください。なお、VSCodeのJupyter Notebookプラグインを使用すると、コンテナにアタッチした後、VSCodeからでも簡単に実行することができます。

## ディレクトリ構成

このリポジトリのディレクトリ構成は以下のようになっています。

```
├── test
│   └── simple_agent.py
├── weights_VAGOsolutions
│   ├── config.json
│   ├── generation_config.json
│   ├── model-00001-of-00002.safetensors
│   ├── model-00002-of-00002.safetensors
│   ├── model.safetensors.index.json
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── tokenizer.json
├── Dockerfile
├── english_nouns_num.pkl
├── english_nouns.pkl
├── large_words_frequency.pickle
├── large_words.pickle
├── main.py
├── pip_freeze.txt
└── requirements.txt
└── test_game.ipynb

```

* **`test` ディレクトリ:** テストゲームをする際の対戦相手となる単純なエージェント (`simple_agent.py`) が格納されています。
* **`weights_VAGOsolutions` ディレクトリ:** 質問者戦略に使用する LLM の重みと設定ファイルが格納されています。
* **`Dockerfile`:** Docker イメージを作成するためのファイルです。
* **`english_nouns_num.pkl`, `english_nouns.pkl`, `large_words_frequency.pickle`, `large_words.pickle`:** 質問者戦略で使用するキーワードリストと単語頻度データです。
* **`main.py`:** 実際に 5 位を獲得したエージェントが定義されています。
* **`pip_freeze.txt`:** `pip freeze` の出力が保存されています。
* **`requirements.txt`:** `main.py` で使用するライブラリが記述されています。
* **`test_game.ipynb`:** `main.py` のエージェントを実際に試すことができます。

## 考慮事項

* 私の実行環境には、大量のメモリ (503 GB) と強力な GPU (4x Quadro RTX 8000) が積んでありますが、kaggle の実行環境で実行可能です。

## Software

pip freeze の出力結果は以下のテキストファイルに保存されています。

* `pip freeze.txt`

## main.py解説

* **質問者エージェント:** 質問者エージェントは、二分探索決定木的なアルゴリズムを使用しています。また、キーワードリストが少なくなってきたらハフマン決定木的なアルゴリズムに変更しています。これによって単語の頻度分析も取り入れ、一般的な単語を優先することで推測精度を向上させています。これは、一般的な単語がシークレットキーワードとして頻繁に登場するという期待に基づいています。

    * **二分探索アルゴリズム:** 
        * 全エージェントに対して、最初のハンドシェイクなしで、辞書順に基づく二分探索を行います。つまり二分探索決定木のような挙動を示します。
        * 残りのキーワード数が少なくなった場合 (全キーワードの長さの合計が約 1400 文字以下になった場合)、辞書順に基づく二分探索から単語頻度に基づく二分探索に切り替えます。つまり、完全には一致しませんがハフマン決定木のような挙動を示します。
        * 単語頻度に基づく二分探索では、以下のような質問を行います。
            "Is the keyword one of the following? {単語頻度上位1位の単語}, {単語頻度上位2位の単語}, {単語頻度上位3位の単語}, …"

    * **キーワードリスト:**
        * 現実世界における具体的な実体を表す特定の名詞で構成されています。
        * 画像キャプションデータセット ([conceptual-captions](https://github.com/google-research-datasets/conceptual-captions), [coco-captions](https://huggingface.co/datasets/sentence-transformers/coco-captions)) と [AmazonReviews](https://github.com/hyp1231/AmazonReviews2023)から名詞のみを抽出し、[gemini flash](https://ai.google.dev/aistudio?hl=ja) を使用して必要な名詞をフィルタリングしています。
        * [wordnet](https://wordnet.princeton.edu/) からの名詞もリストに追加しています。
        * キーワードリストに対して、単語頻度を計算しています。[wordfreq](https://pypi.org/project/wordfreq/)

* **回答者エージェント:** 回答者エージェントは、キーワードの属性に関する質問に対して、LLM (llama3.1) ベースのアプローチを採用しています。

    * **使用している言語モデル:**  Hugging Face Hub で公開されている [`VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct`](https://huggingface.co/VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct) モデルを使用しています。これは、オープンリーダーボードで高い iFEVal スコアを持つ、Llama 3.1 ベースの 8B モデルです。
    * **質問解釈と回答生成:** 
        * 文字構造に関する質問 (例: "キーワードは 'a' で始まりますか?") に対しては、`main.py` の `simple_question` 関数で実装されている正規表現を用いたハードコーディングで回答を生成しています。
        * 属性に関する質問 (例: "キーワードは動物ですか?") に対しては、llama3.1 モデルを使用して回答を生成しています。

* **推論エージェント:** 推論エージェントは、現在のワードリストの中で最も単語頻度の高い単語を推測キーワードとして選択しています。