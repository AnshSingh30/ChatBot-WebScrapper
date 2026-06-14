# Website Chatbot — How to Run

Console-based chatbot that crawls a website, indexes its content, and answers questions using the **Google Gemini API**.

**Reference site:** https://botpenguin.com/

---

## Prerequisites

- Python 3.8 or higher
- A Google Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

---

## Setup

### 1. Download the project

Place all project files in a folder on your machine.

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install requests beautifulsoup4 python-dotenv
```

### 4. Configure your API key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

**Do not share or commit your API key.**

### 5. Optional settings (in `.env`)

These are optional. Defaults work if you omit them.

| Variable | Description | Example |
|----------|-------------|---------|
| `USE_KEYWORD_MODE` | Skip embeddings to save API quota | `true` |
| `MAX_CRAWL_PAGES` | Max pages to crawl | `200` |
| `MAX_CRAWL_DEPTH` | Max link depth from start URL | `8` |
| `REFRESH_CACHE` | Re-crawl and rebuild index | `true` / `false` |
| `CRAWL_DELAY_SECONDS` | Delay between page requests | `0.5` |

**Recommended for first run:**

```env
GOOGLE_API_KEY=your_gemini_api_key_here
USE_KEYWORD_MODE=true
MAX_CRAWL_PAGES=200
MAX_CRAWL_DEPTH=8
REFRESH_CACHE=true
CRAWL_DELAY_SECONDS=0.5
```

After the first successful run, set `REFRESH_CACHE=false` to reuse the cached index in `.cache/`.

---

## Run the chatbot

```bash
python chatbot.py
```

### What happens

1. You are prompted to enter a website URL (e.g. `https://botpenguin.com/`).
2. The bot crawls pages on that site and builds a knowledge index.
3. You can ask questions about the website content in the console.
4. Type `exit` to quit.

### Example session

```
Enter website URL: https://botpenguin.com/

Crawling site (max 200 pages, depth 8)...
Indexed 150 page(s), 45000 words, 120 chunk(s).

Chatbot is ready for: https://botpenguin.com/
Ask any question about the website content.

You: What is BotPenguin?
Bot: ...

You: exit
Goodbye!
```

---

## Troubleshooting

### API rate limit (HTTP 429)

If you hit Gemini quota limits:

- Set `USE_KEYWORD_MODE=true` in `.env`
- Reduce crawl size: `MAX_CRAWL_PAGES=30`
- Increase delays: `EMBED_DELAY_SECONDS=10`
- Reuse cache: `REFRESH_CACHE=false`

### Missing API key

```
Error: GOOGLE_API_KEY is not set.
```

Create a `.env` file with your Gemini API key.

### Slow first run

Crawling 200 pages can take several minutes. This is normal. Later runs are faster if `REFRESH_CACHE=false`.

---

## Project files

| File | Purpose |
|------|---------|
| `chatbot.py` | Main chatbot script (includes step-by-step process in docstring) |
| `requirements.txt` | Python dependencies |
| `.env` | Your API key and settings (create this yourself) |
| `.cache/` | Cached index (auto-created on first run) |

---

## Notes for reviewers

- The step-by-step development process is documented at the top of `chatbot.py`.
- No frontend is required; the chatbot runs entirely in the console.
- Scraping uses `requests` + BeautifulSoup; only question answering uses the Gemini API when `USE_KEYWORD_MODE=true`.
