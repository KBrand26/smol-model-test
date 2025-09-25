## SmolLM2 via Ollama – Terminal Chat Harness

This repo contains a tiny Python script to chat with SmolLM2 (1.7B) locally via the Ollama HTTP API. It supports interactive chat, one-shot prompts, streaming output, simple commands, and auto-pulling the model if needed.

Model reference: [SmolLM2 on Ollama](https://ollama.com/library/smollm2)

### Prerequisites
- macOS, Linux, or Windows with WSL
- Python 3.13+
- Ollama installed and the daemon running
  - Install: see `https://ollama.com`
  - Run daemon: `ollama serve`

### Install
No extra Python deps required. Use the system Python (3.13+) or create a venv.

### Usage
Interactive chat:

```bash
python main.py
```

One-shot prompt:

```bash
python main.py -p "Explain what SmolLM2 is in 2 sentences."
```

Specify a different model or options:

```bash
python main.py \
  --model smollm2:1.7b \
  --system "You are a concise assistant." \
  --temperature 0.3 \
  --ctx 8192 \
  --max-tokens 512
```

Environment overrides:

- `OLLAMA_HOST` (default `127.0.0.1`)
- `OLLAMA_PORT` (default `11434`)
- `OLLAMA_MODEL` (default `smollm2:1.7b`)

### Interactive commands
- `/help` – show commands
- `/reset` – clear conversation history (keeps system prompt if provided)
- `/save <path>` – save transcript to a JSON file
- `/exit` or `/quit` – leave

### Notes
- On first use, the script will try to pull `smollm2:1.7b` via the `ollama` CLI if it is available; otherwise, it will print instructions to install/pull manually.
- Make sure the Ollama daemon is running before use: `ollama serve`

### Reference
- SmolLM2 on Ollama: `https://ollama.com/library/smollm2`
