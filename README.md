<div align="center">

# 🔍 RagScope

### Advanced RAG Techniques \& Evaluation Framework

Build, compare and **measure** advanced retrieval strategies, side by side.

!\[Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python\&logoColor=white)
!\[LlamaIndex](https://img.shields.io/badge/LlamaIndex-RAG-7C3AED)
!\[TruLens](https://img.shields.io/badge/TruLens-Evals-0EA5E9)
!\[Groq](https://img.shields.io/badge/LLM-Groq-F55036)

</div>

\---

## 📖 Overview

Most RAG demos stop at "it returns an answer". **RagScope** goes
further: it implements two advanced retrieval strategies and scores
them with an automated evaluation pipeline, so you can see *which
approach actually performs better* on your own documents.

|||
|-|-|
|🧩 **Retrievers**|Sentence Window, Auto-Merging|
|🎯 **Reranking**|Cross-encoder on both pipelines|
|📊 **Evaluation**|TruLens RAG Triad|
|📈 **Output**|Interactive comparison dashboard|

\---

## ✨ Key Features

* **Two advanced retrievers** under one runner, using the same
documents, models and questions for a fair comparison.
* **Cross-encoder reranking** to sharpen the top results before the
LLM sees them.
* **RAG Triad evaluation** with LLM-as-judge reasoning:
answer relevance, context relevance and groundedness.
* **Batch evaluation** from a plain text file, one question per line.
* **Live dashboard** to compare versions, scores and traces.
* **Deterministic answers** with `temperature=0` for reproducible runs.
* **Modular code**: each retriever and the evals live in their own file.

\---

## 🏗️ Architecture

RagScope runs **two advanced RAG pipelines on the same documents and evaluation questions**, then sends both outputs to TruLens for a fair comparison.

!\[RagScope Architecture](ragscope\_readme\_assets/ragscope\_architecture.png)

### Retrieval strategies at a glance

!\[Sentence Window and Auto-Merging Retrieval](ragscope\_readme\_assets/retrieval\_strategies.png)

## 🧠 Retrieval Strategies

### 1\. Sentence Window Retrieval

Documents are split into **single sentences** for precise embedding
and matching. At query time, each matched sentence is expanded back
to its surrounding window, so the LLM gets richer context than the
sentence alone.

|Setting|Value|
|-|-|
|Window size|4 sentences|
|Similarity top-k|6|
|Rerank top-n|3|
|Postprocessor|Metadata replacement|

### 2\. Auto-Merging Retrieval

Documents are split into a **three-level hierarchy**. Only the small
leaf chunks are embedded. When enough sibling leaves are retrieved,
they are **merged into their parent chunk**, giving the LLM a more
coherent passage.

|Setting|Value|
|-|-|
|Chunk sizes|1024 / 256 / 64|
|Similarity top-k|6|
|Merge threshold|0.3|
|Rerank top-n|2|

> 💡 \*\*Rule of thumb:\*\* Sentence Window favors precision.
> Auto-Merging favors coherence. RagScope helps you verify that on
> your own data.

\---

## 📊 Evaluation Framework

RagScope scores every question with the **RAG Triad**, using an LLM
judge that also returns its reasoning.

|Metric|Question it answers|
|-|-|
|**Answer Relevance**|Does the answer address the question?|
|**Context Relevance**|Is the retrieved context useful?|
|**Groundedness**|Is the answer supported by the context?|

Together they help you spot the three classic failure modes:
irrelevant answers, poor retrieval and hallucination.

\---

## 🧰 Tech Stack

|Layer|Technology|
|-|-|
|Orchestration|LlamaIndex|
|LLM|Groq `openai/gpt-oss-120b`|
|Embeddings|`BAAI/bge-small-en-v1.5`|
|Reranker|`cross-encoder/ms-marco-MiniLM-L-6-v2`|
|Evaluation|TruLens + LiteLLM|
|Numerics|NumPy|

\---

## 📁 Project Structure

```text
RagScope/
├── main.py
├── sentence\_window\_retriever.py
├── automerging\_retriever.py
├── Evaluations.py
├── Rag\_Eval.txt
├── data/
│   └── \*.txt
└── README.md
```

|File|Purpose|
|-|-|
|`main.py`|Runs both pipelines, evals, dashboard|
|`sentence\_window\_retriever.py`|Nodes, index, query engine|
|`automerging\_retriever.py`|Hierarchy, index, query engine|
|`Evaluations.py`|TruLens metric definitions|
|`Rag\_Eval.txt`|Evaluation questions|
|`data/`|Your `.txt` source documents|

\---

## 🚀 Getting Started

### 1\. Clone the repository

```bash
git clone https://github.com/<your-username>/RagScope.git
cd RagScope
```

### 2\. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

On Windows, use `venv\\Scripts\\activate`.

### 3\. Install dependencies

```bash
pip install llama-index
pip install llama-index-llms-groq
pip install llama-index-embeddings-huggingface
pip install sentence-transformers numpy
pip install trulens trulens-apps-llamaindex
pip install trulens-providers-litellm
pip install trulens-dashboard
```

### 4\. Set your Groq API key

```bash
export GROQ\_API\_KEY="your\_key\_here"
```

On Windows PowerShell:

```powershell
$env:GROQ\_API\_KEY="your\_key\_here"
```

### 5\. Add your data

* Put your `.txt` documents inside `data/`.
* Add one evaluation question per line in `Rag\_Eval.txt`.

```text
What is RAG?
How does reranking improve retrieval?
Why use sentence window retrieval?
```

### 6\. Run

```bash
python main.py
```

\---

## ▶️ What Happens When You Run It

1. Loads documents from `data/`.
2. Builds the Sentence Window and Auto-Merging pipelines.
3. Prints a sample answer from the Sentence Window engine.
4. Reads every question from `Rag\_Eval.txt`.
5. Runs each question through both engines, recorded by TruLens.
6. Launches the dashboard to compare results.

\---

## 📈 Reading the Dashboard

* Open the local URL printed in your terminal.
* Compare **Sentence Window RAG** and **Automerging RAG** by app
name and version.
* Open a record to inspect retrieved context, the final answer and
the judge's reasoning.
* Look for low **Groundedness** scores first. They point to
hallucination risk.

\---

## 🔧 Configuration

Tune these values to fit your corpus.

|Parameter|Where|Effect|
|-|-|-|
|`window\_size`|`sentence\_window\_retriever.py`|Context around each sentence|
|`chunk\_sizes`|`automerging\_retriever.py`|Hierarchy granularity|
|`simple\_ratio\_thresh`|`automerging\_retriever.py`|How eagerly chunks merge|
|`similarity\_top\_k`|`main.py`|Candidates retrieved|
|`top\_n`|`main.py`|Candidates kept after rerank|

\---

## 🗺️ Roadmap

* \[ ] Support PDF and Markdown ingestion
* \[ ] Add a baseline (naive chunking) pipeline
* \[ ] Export evaluation results to CSV
* \[ ] Add hybrid (BM25 + vector) retrieval
* \[ ] Add a config file instead of hard-coded values
* \[ ] Dockerize the project

\---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit your changes
4. Open a pull request

\---

## 📄 License

Distributed under the MIT License. Add a `LICENSE` file to the
repository root to make this official.

\---

<div align="center">

**If RagScope helped you, consider giving it a ⭐**

</div>

