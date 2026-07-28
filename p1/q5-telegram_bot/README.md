# Telegram Data Analyst

Webhook-first Telegram bot that downloads public tabular datasets, performs deterministic pandas analysis, and uses AI Pipe only for conversational reasoning. Each normal reply is exactly one JSON object with an answer and a public JSONL audit-log URL.

## Run locally

1. Copy `.env.example` to `.env` and supply AI Pipe, Telegram, and public HTTPS URL values.
2. Install Python 3.12 dependencies: `pip install -r requirements-dev.txt`.
3. Start: `uvicorn app:app --host 0.0.0.0 --port 8000`.
4. Set the Telegram webhook after deployment:
   `https://api.telegram.org/bot<TOKEN>/setWebhook?url=<PUBLIC_URL>/webhook`

The public URL must be HTTPS because Telegram webhooks require it. Docker deployment is available with `docker compose up --build`; Render is configured through `render.yaml`.

## Grader contract

Every normal-message reply is serialized as exactly `{"answer": <requested shape>, "log_url": <public JSONL URL>}`. The agent recognises an example JSON shape embedded in the question (such as `{"answer":{"state":"<state name>"}}`) and fills that inner shape rather than returning a prose answer. Multi-turn context is kept per Telegram user, and the final message is answered independently with the preceding turns as reasoning context.

The supplied public evaluator clone is useful for delivery testing: it sends each turn and evaluates its final response. Its current sample `grade.py` compares replies to the inner object directly, whereas the project brief requires the outer answer/log URL envelope; use the brief's envelope for deployment unless the course staff publish an updated evaluator contract.

## Supported data and questions

CSV/TSV, Excel, JSON, HTML tables, ZIP archives containing a supported file, and extension-less public CSV endpoints are supported. Ask for row count, columns, missing values, mean, median, mode, sum, standard deviation, count, unique values, sorting, and top/bottom N for an explicitly named column. For example: `What is the mean of revenue? https://example.com/sales.csv`.

Operational endpoints: `/`, `/health`, `/status`, `/webhook`, `/logs/{run_id}.jsonl`, and FastAPI `/docs`.
