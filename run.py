import argparse
import importlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
EXTRAS_MAP_PATH = PROJECT_ROOT / "provider_extras.toml"

ROLE_ENV_VARS = {
    "llm": "LLM_PROVIDER",
    "embedding": "EMBEDDING_PROVIDER",
    "query_rewrite": "QUERY_REWRITE_PROVIDER",
    "reranker": "RERANKER_PROVIDER",
}

PROVIDER_IMPORT_CHECKS = {
    ("llm", "local"): "llama_index.llms.ollama",
    ("llm", "openai"): "llama_index.llms.openai",
    ("llm", "gemini"): "llama_index.llms.google_genai",
    ("llm", "claude"): "llama_index.llms.anthropic",
    ("embedding", "local"): "llama_index.embeddings.ollama",
    ("embedding", "openai"): "llama_index.embeddings.openai",
    ("embedding", "gemini"): "llama_index.embeddings.google_genai",
    ("embedding", "voyage"): "llama_index.embeddings.voyageai",
    ("query_rewrite", "local"): "llama_index.llms.ollama",
    ("query_rewrite", "openai"): "llama_index.llms.openai",
    ("query_rewrite", "gemini"): "llama_index.llms.google_genai",
    ("query_rewrite", "claude"): "llama_index.llms.anthropic",
    ("reranker", "local"): "sentence_transformers",
    ("reranker", "cohere"): "llama_index.postprocessor.cohere_rerank",
}

def _load_extras_map() -> dict:
    with open(EXTRAS_MAP_PATH, "rb") as f:
        return tomllib.load(f)


def _load_env_values() -> dict:
    if ENV_PATH.exists():
        return dotenv_values(ENV_PATH)
    return {}

def _missing_providers(env_values: dict) -> list:
    missing = []
    for role, env_var in ROLE_ENV_VARS.items():
        provider = env_values.get(env_var)
        if not provider:
            continue  
        module_name = PROVIDER_IMPORT_CHECKS.get((role, provider))
        if module_name is None:
            continue 
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append((role, provider, module_name))
    return missing

def _run_setup_wizard(non_interactive: bool) -> bool:
    cmd = [sys.executable, str(PROJECT_ROOT / "src" / "setup_wizard.py")]
    if non_interactive:
        cmd.append("--non-interactive")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0

def _run_streamlit():
    cmd = ["streamlit", "run", str(PROJECT_ROOT / "ui" / "app.py")]
    os.execvp(cmd[0], cmd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="If setup is needed, run it non-interactively (for Docker/CI).",
    )
    args = parser.parse_args()

    env_values = _load_env_values()

    if not ENV_PATH.exists() or not all(
        env_values.get(v) for v in ROLE_ENV_VARS.values()
    ):
        print("Provider configuration incomplete -- launching setup wizard.\n")
        if not _run_setup_wizard(args.non_interactive):
            print("Setup failed. Exiting.")
            sys.exit(1)
        env_values = _load_env_values()  # reload after wizard wrote .env

    missing = _missing_providers(env_values)
    if missing:
        print("The following selected providers are missing their SDK:")
        for role, provider, module_name in missing:
            print(f"  - {role}: '{provider}' (needs to import '{module_name}')")
        print("\nLaunching setup wizard to install them.\n")
        if not _run_setup_wizard(args.non_interactive):
            print("Setup failed. Exiting.")
            sys.exit(1)

    print("All selected providers are ready. Starting the app...\n")
    _run_streamlit()


if __name__ == "__main__":
    main()