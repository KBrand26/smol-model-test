import argparse
import http.client
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("OLLAMA_PORT", "11434"))
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "smollm2:1.7b")

def print_err(message: str) -> None:
    print(message, file=sys.stderr, flush=True)

def is_ollama_running(host: str, port: int) -> bool:
    try:
        conn = http.client.HTTPConnection(host, port, timeout=2)
        conn.request("GET", "/api/version")
        resp = conn.getresponse()
        return resp.status == 200
    except Exception:
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass

def ensure_model_available(model: str) -> None:
    """Ensure the model is available locally; attempt to pull if missing.

    Uses the `ollama` CLI if present. If not available, prints a hint.
    """
    ollama_cli = shutil.which("ollama")
    if not ollama_cli:
        print_err("'ollama' CLI not found in PATH. Skipping auto-pull.")
        print_err("Install Ollama and ensure the daemon is running: https://ollama.com")
        return

    try:
        check = subprocess.run(
            [ollama_cli, "show", model],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if check.returncode == 0:
            return
    except Exception:
        pass

    print_err(f"Pulling model '{model}' via ollama... This may take a while on first run.")
    try:
        pull = subprocess.run([ollama_cli, "pull", model], check=False)
        if pull.returncode != 0:
            print_err("Failed to pull model via ollama CLI. You can pull manually:")
            print_err(f"  ollama pull {model}")
    except FileNotFoundError:
        print_err("'ollama' CLI not found in PATH. Install from https://ollama.com")


def stream_chat(
    host: str,
    port: int,
    model: str,
    messages: List[Dict[str, str]],
    options: Optional[Dict] = None,
    stream: bool = True,
) -> Dict:
    """Send a chat request to Ollama and stream tokens as they arrive.

    Returns the final response dict from Ollama (the last JSON object with done=true).
    """
    body = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if options:
        body["options"] = options

    conn = http.client.HTTPConnection(host, port, timeout=60)
    try:
        conn.request(
            "POST",
            "/api/chat",
            body=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        if resp.status != 200:
            data = resp.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama error {resp.status}: {data}")

        final: Dict = {}
        buffer = b""
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue

                message = event.get("message") or {}
                content = message.get("content", "")
                if content:
                    print(content, end="", flush=True)

                if event.get("done"):
                    final = event
                    print()
                    return final

        return final
    finally:
        try:
            conn.close()
        except Exception:
            pass


def interactive_chat(
    model: str,
    system_prompt: Optional[str],
    temperature: Optional[float],
    num_ctx: Optional[int],
    num_predict: Optional[int],
    no_stream: bool,
) -> None:
    if not is_ollama_running(DEFAULT_HOST, DEFAULT_PORT):
        print_err(
            f"Cannot reach Ollama at http://{DEFAULT_HOST}:{DEFAULT_PORT}. Is the daemon running?"
        )
        print_err("Start it with: ollama serve")
        return

    ensure_model_available(model)

    history: List[Dict[str, str]] = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})

    options: Dict = {}
    if temperature is not None:
        options["temperature"] = temperature
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    if num_predict is not None:
        options["num_predict"] = num_predict

    print(f"Model: {model}")
    print("Type '/help' for commands. Start chatting.\n")

    while True:
        try:
            user = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user:
            continue

        if user in {"/exit", "/quit"}:
            break
        if user == "/help":
            print(
                "Commands:\n"
                "  /exit or /quit   Exit the chat\n"
                "  /reset           Reset the conversation history\n"
                "  /save <path>     Save transcript to a JSON file\n"
                "  /help            Show this help\n"
            )
            continue
        if user == "/reset":
            history = [{"role": "system", "content": system_prompt}] if system_prompt else []
            print("History reset.")
            continue
        if user.startswith("/save"):
            _, _, path = user.partition(" ")
            path = path.strip() or f"chat_{int(time.time())}.json"
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"model": model, "history": history}, f, ensure_ascii=False, indent=2)
                print(f"Saved transcript to {path}")
            except Exception as e:
                print_err(f"Failed to save transcript: {e}")
            continue

        history.append({"role": "user", "content": user})
        print("Assistant>", end=" ", flush=True)
        try:
            resp = stream_chat(
                host=DEFAULT_HOST,
                port=DEFAULT_PORT,
                model=model,
                messages=history,
                options=options,
                stream=not no_stream,
            )
            msg = (resp.get("message") or {}).get("content", "") if isinstance(resp, dict) else ""
            history.append({"role": "assistant", "content": msg})
        except Exception as e:
            print_err(f"\nError: {e}")


def one_shot(
    prompt: str,
    model: str,
    system_prompt: Optional[str],
    temperature: Optional[float],
    num_ctx: Optional[int],
    num_predict: Optional[int],
    no_stream: bool,
) -> None:
    if not is_ollama_running(DEFAULT_HOST, DEFAULT_PORT):
        print_err(
            f"Cannot reach Ollama at http://{DEFAULT_HOST}:{DEFAULT_PORT}. Is the daemon running?"
        )
        print_err("Start it with: ollama serve")
        sys.exit(1)

    ensure_model_available(model)

    history: List[Dict[str, str]] = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})
    history.append({"role": "user", "content": prompt})

    options: Dict = {}
    if temperature is not None:
        options["temperature"] = temperature
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    if num_predict is not None:
        options["num_predict"] = num_predict

    try:
        stream_chat(
            host=DEFAULT_HOST,
            port=DEFAULT_PORT,
            model=model,
            messages=history,
            options=options,
            stream=not no_stream,
        )
    except Exception as e:
        print_err(f"Error: {e}")
        sys.exit(1)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Terminal chat harness for Ollama SmolLM2")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Model name, e.g. smollm2:1.7b")
    parser.add_argument("-s", "--system", default=None, help="Optional system prompt")
    parser.add_argument("-t", "--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--ctx", type=int, default=None, help="Context window size (num_ctx)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens to generate (num_predict)")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    parser.add_argument("-p", "--prompt", default=None, help="One-shot prompt (non-interactive)")
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    if args.prompt:
        one_shot(
            prompt=args.prompt,
            model=args.model,
            system_prompt=args.system,
            temperature=args.temperature,
            num_ctx=args.ctx,
            num_predict=args.max_tokens,
            no_stream=args.no_stream,
        )
    else:
        interactive_chat(
            model=args.model,
            system_prompt=args.system,
            temperature=args.temperature,
            num_ctx=args.ctx,
            num_predict=args.max_tokens,
            no_stream=args.no_stream,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
