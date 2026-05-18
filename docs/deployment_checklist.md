# Deployment Checklist

Use this checklist before running the FastAPI essay rater service on a new machine.

## Services

- Bareun Docker container is running and reachable on port `5656`.
- `BAREUN_API_KEY` is set in `.env`.
- `MODEL_PATH` points to the base model path or Hugging Face model ID.
- LoRA adapter files are available under `rater/` or the configured adapter path.

## Python environment

```bash
pip install -r requirements.txt
```

Install vLLM separately according to the CUDA version on the target machine.

## Smoke test

```bash
python app.py
```

Open `http://localhost:8000` and submit a short essay to confirm SSE streaming, feature extraction, and final scoring all complete.

## Runtime artifacts

Keep `.env`, CA bundle files, model weights, adapter backups, and logs out of Git.
