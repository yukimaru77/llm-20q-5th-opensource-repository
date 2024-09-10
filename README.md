# LLM 20 Questions - yukky_maru

This repository contains the agent for the LLM 20 Questions competition along with the code and files necessary to run it. However, the LLM weights used by the agent cannot be uploaded to GitHub, so you need to download them manually and modify the `config.json` file. Therefore, I recommend using the Docker image that I have uploaded to Dockerhub.

## Video Explanation

I have created a video explaining how to pull the Docker image and test the llm 20 question with my agent. Please refer to it.
https://youtu.be/D77itV-L44s

## Requirements

To use the code in this repository, you need the official Kaggle environment `gcr.io/kaggle-gpu-images/python`, additional Python libraries, and the LLM weights used by the agent. The necessary items are as follows:

* Kaggle environment
* **Additional Python Libraries:**
    * torch==2.1.2
    * transformers==4.42.3
    * bitsandbytes==0.43.3
    * accelerate==0.33.0
* LLM weights and configuration files (note: some configuration files need to be modified)

### Using Dockerhub

1. Run the following command to download the Docker image I uploaded to Dockerhub:

```bash
docker pull yukimaru77/llm-20-questions_5th:latest
```

### If not using Dockerhub

1. Clone this repository.
2. Create a directory named `weights_VAGOsolutions` in the same directory as `main.py`, and store the LLM weights from [here](https://huggingface.co/VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct) in that directory.
3. Modify the `rope_scaling` section of the `config.json` file inside `weights_VAGOsolutions` as follows:

```json
  "rope_scaling": {
    "factor": 8.0,
    "type": "dynamic"
  }
```
4. Build the Dockerfile:

```bash
docker build -t llm-20-questions_5th -f Dockerfile .
```

## Environment Setup

1. Run the Docker container:

```bash
docker run -itd --gpus all --name llm-20-questions_5th -p 8889:8888 yukimaru77/llm-20-questions_5th:latest jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
```

## Hardware

My environment:

* **CPU:** AMD EPYC 7702P 64-Core Processor
* **Memory:** 503 GB
* **GPU:** 4x Quadro RTX 8000

## OS

* **Ubuntu 20.04.6 LTS (Focal Fossa)**

## Prediction

The agent function necessary to run the game, the keyword list, and the model weights are stored in the following locations:

* Agent function: `main.py`
* Keyword list: `english_nouns_num.pkl`, `english_nouns.pkl`, `large_words_frequency.pickle`, `large_words.pickle`
* Model weights: `weights_VAGOsolutions`

## Testing

You can use `test_game.ipynb` to test my agents. The necessary code is stored under the `/app` directory. If you want to run the test, please execute `test_game.ipynb`. Additionally, by using the Jupyter Notebook plugin in VSCode, you can easily execute it from VSCode after attaching to the container.

## Directory Structure

The directory structure of this repository is as follows:

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

* **`test` directory:** Contains a simple agent (`simple_agent.py`) that acts as the opponent during test games.
* **`weights_VAGOsolutions` directory:** Contains the LLM weights and configuration files used for the questioner's strategy.
* **`Dockerfile`:** File used to create the Docker image.
* **`english_nouns_num.pkl`, `english_nouns.pkl`, `large_words_frequency.pickle`, `large_words.pickle`:** Keyword lists and word frequency data used in the questioner strategy.
* **`main.py`:** Defines the agent that achieved 5th place.
* **`pip_freeze.txt`:** Contains the output of the `pip freeze` command.
* **`requirements.txt`:** Lists the libraries used in `main.py`.
* **`test_game.ipynb`:** Used to test the agent defined in `main.py`.

## Considerations

My environment has a large amount of memory (503 GB) and powerful GPUs (4x Quadro RTX 8000), but this can be run in the Kaggle environment.

## Software

The output of `pip freeze` is saved in the following text file:

* `pip freeze.txt`

## Explanation of `main.py`

* **Questioner Agent:** The questioner agent uses a binary search decision tree-like algorithm. When the keyword list becomes smaller, it switches to a Huffman decision tree-like algorithm. By incorporating word frequency analysis, the agent prioritizes common words to improve prediction accuracy. This is based on the expectation that common words are more likely to be the secret keyword.

    * **Binary Search Algorithm:** 
        * The agent conducts a binary search based on dictionary order without an initial handshake for all agents, behaving similarly to a binary search decision tree.
        * When the total length of the remaining keyword list becomes less than about 1400 characters, the search switches from dictionary-order-based binary search to frequency-based binary search, behaving similarly to a Huffman decision tree.
        * In frequency-based binary search, the agent asks questions like:
            "Is the keyword one of the following? {top frequency word}, {second top frequency word}, {third top frequency word}, …"

    * **Keyword List:**
        * Composed of specific nouns representing tangible entities in the real world.
        * Nouns were extracted from image caption datasets ([conceptual-captions](https://github.com/google-research-datasets/conceptual-captions), [coco-captions](https://huggingface.co/datasets/sentence-transformers/coco-captions)) and [AmazonReviews](https://github.com/hyp1231/AmazonReviews2023), and filtered using [gemini flash](https://ai.google.dev/aistudio?hl=ja) to obtain the necessary nouns.
        * Nouns from [WordNet](https://wordnet.princeton.edu/) were also added to the list.
        * Word frequencies were calculated for the keyword list using [wordfreq](https://pypi.org/project/wordfreq/).

* **Answerer Agent:** The answerer agent uses an LLM (Llama 3.1) approach to answer questions about the attributes of the keyword.

    * **Language Model Used:** Uses the [`VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct`](https://huggingface.co/VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct) model, which is based on Llama 3.1 with 8B parameters and holds a high iFEVal score on the open leaderboard.
    * **Question Interpretation and Answer Generation:** 
        * For questions about letter structure (e.g., "Does the keyword start with 'a'?"), answers are generated using hard-coded regular expressions implemented in the `simple_question` function in `main.py`.
        * For questions about attributes (e.g., "Is the keyword an animal?"), the Llama 3.1 model is used to generate answers.

* **Inference Agent:** The inference agent selects the keyword with the highest word frequency from the remaining word list as the guessed keyword.