# Finance Classifier V2

Desktop app that estimates **financial sentiment** (negative / neutral / positive) for news-style articles. Paste a BENZINGA URL, fetch the page, split the body into sentences, run each sentence through a small transformer classifier, then combine scores and show the result in a **CustomTkinter** UI.

## Features

- **URL input** — Fetches HTML with `requests`, strips boilerplate with **BeautifulSoup** (paragraphs only; removes common chrome like `nav`, `footer`, etc.).
- **Sentence-level inference** — **NLTK** `sent_tokenize`, then **Hugging Face Transformers** with a 3-class finance sentiment model.
- **Aggregate score** — Logits are summed over sentences, then **softmax** gives one distribution for the whole article.
- **Progress bars** — Red / yellow / green bars for negative, neutral, and positive share (values in 0–1 for `CTkProgressBar`).

## Model

- **Checkpoint:** `[Mysticz0/finance-pro-model-v1.0](https://huggingface.co/Mysticz0/finance-pro-model-v1.0)` on the Hugging Face Hub.
- First launch downloads model weights (network required once, unless cached).

## Requirements

- **Python** 3.10+ recommended  
- Dependencies (install explicitly; there is no bundled `requirements.txt` in this repo):

```text
torch
transformers
customtkinter
nltk
requests
beautifulsoup4
```

Example:

```bash
pip install torch transformers customtkinter nltk requests beautifulsoup4
```

On first use, NLTK may download `**punkt_tab**` for sentence tokenization (handled in code if missing).

## Run

From the project directory (so `logo.ico` resolves):

```bash
python main.py
```



The window title is **Financial Sentiment Analyzer**. Enter an article URL and click **Analyze**. While work runs, the button shows **Loading..** 

## Project layout


| File       | Role                         |
| ---------- | ---------------------------- |
| `main.py`  | Full app: scrape, model, GUI |
| `logo.ico` | Window icon                  |


## Limitations

- **Extraction** — HTML parsing is heuristic; some sites will yield poor or empty text. Trailing junk is cut at fixed phrases (e.g. photo/disclaimer markers).
- **Main thread** — Fetch and inference run on the UI thread; long articles can freeze the window until processing finishes.
- **Remote pages** — No paywall bypass; robots and rate limits apply as for any simple HTTP client.

