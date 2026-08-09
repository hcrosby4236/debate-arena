# Debate Arena

Debate Arena is a Flask single-page web app that runs a multi-agent debate using the Backboard API.

## Requirements

- Python 3.10+
- A Backboard API key in the `BACKBOARD_API_KEY` environment variable

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export BACKBOARD_API_KEY="your-api-key"
python app.py
```

Open <http://127.0.0.1:5000>.

If your Python install blocks global pip installs and cannot create virtual environments, install into a local target instead:

```bash
python3 -m pip install --target .deps -r requirements.txt
export BACKBOARD_API_KEY="your-api-key"
PYTHONPATH=.deps python3 app.py
```

## How it works

On the first run, the app creates three Backboard assistants and saves their IDs to `agents.json`:

- `PRO AGENT` researches and argues for the submitted question with `web_search="Auto"`.
- `CONS AGENT` researches and argues against the submitted question with `web_search="Auto"`.
- `JUDGE` reviews both sides and chooses the morally stronger winner.

Later runs reuse the assistant IDs in `agents.json`.

## API

You can also run a debate from the command line while the server is running:

```bash
curl -s -X POST http://127.0.0.1:5000/api/debate \
  -H 'Content-Type: application/json' \
  -d '{"topic":"Are AI Agents necessary?"}'
```
