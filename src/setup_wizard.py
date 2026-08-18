import argparse
import os
import subprocess
import sys
import tomllib
from pathlib import Path

from questionary import Choice
import questionary
from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
ENV_EXAMPLE_PATH = PROJECT_ROOT / ".env.example"
EXTRAS_MAP_PATH = PROJECT_ROOT / "provider_extras.toml"

ROLE_ENV_VARS = {
    "llm": "LLM_PROVIDER",
    "embedding": "EMBEDDING_PROVIDER",
    "query_rewrite": "QUERY_REWRITE_PROVIDER",
    "reranker": "RERANKER_PROVIDER",
}

ROLE_LABELS = {
    "llm": "Main LLM (answers questions)",
    "embedding": "Embedding model (indexes documents)",
    "query_rewrite": "Query rewriter (resolves conversational context)",
    "reranker": "Reranker (re-scores retrieved chunks)",
}

def _load_extras_map() -> dict:
    with open(EXTRAS_MAP_PATH, "rb") as f:
        return tomllib.load(f)

def _load_env_values() -> dict:
    if ENV_PATH.exists():
        return dotenv_values(ENV_PATH)
    if ENV_EXAMPLE_PATH.exists():
        return dotenv_values(ENV_EXAMPLE_PATH)
    return {}

def _write_provider_choices(choices: dict):
    existing_lines = []
    source_path = ENV_PATH if ENV_PATH.exists() else ENV_EXAMPLE_PATH
    if source_path.exists():
        existing_lines = source_path.read_text().splitlines()

    updated_keys = set()
    new_lines = []
    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            new_lines.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in choices:
            new_lines.append(f"{key}={choices[key]}")
            updated_keys.add(key)
        else:
            new_lines.append(line)

    for key, value in choices.items():
        if key not in updated_keys:
            new_lines.append(f"{key}={value}")

    ENV_PATH.write_text("\n".join(new_lines) + "\n")
    print(f"Updated {ENV_PATH}")

def _format_provider_label(role: str, provider: str) -> str:
    if provider != "local":
        return provider
    if role == "reranker":
        return "local(bge-reranker-v2-m3)"
    return "local (Ollama)"

def _extras_for_choices(extras_map: dict, choices: dict) -> list:
    extras = set()
    for role, env_var in ROLE_ENV_VARS.items():
        provider = choices.get(env_var)
        if not provider:
            continue
        role_map = extras_map.get(role, {})
        extra_group = role_map.get(provider)
        if extra_group:
            extras.add(extra_group)
    return sorted(extras)

def _run_uv_sync(extras: list) -> bool:
    if not extras:
        print("No extras needed -- nothing to install.")
        return True

    cmd = ["uv", "sync"]
    for extra in extras:
        cmd += ["--extra", extra]

    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0

def run_interactive():
    extras_map = _load_extras_map()
    current_values = _load_env_values()

    print("GDrive-RAG Provider Setup\n")
    print("Choose a provider for each component. 'local' uses a local provider")
    print("and needs no API key. Cloud providers")
    print("need the matching API key set in .env afterward.\n")

    choices = {}
    for role, env_var in ROLE_ENV_VARS.items():
        available_providers = list(extras_map.get(role, {}).keys())
        current = current_values.get(env_var) or "local"
        default_index = (
            available_providers.index(current)
            if current in available_providers
            else 0
        )

        provider_choices = [
            Choice(_format_provider_label(role, provider), provider)
            for provider in available_providers
        ]
        default_choice = provider_choices[default_index]

        selected = questionary.select(
            ROLE_LABELS[role],
            choices=provider_choices,
            default=default_choice,
            pointer=">",
        ).ask()

        if selected is None: 
            print("\nSetup cancelled.")
            sys.exit(1)

        choices[env_var] = selected

    _write_provider_choices(choices)

    extras = _extras_for_choices(extras_map, choices)
    success = _run_uv_sync(extras)

    if not success:
        print("\n`uv sync` failed -- see the error above.")
        sys.exit(1)

    cloud_providers_selected = {v for v in choices.values() if v != "local"}
    if cloud_providers_selected:
        print(
            "\nReminder: you selected cloud provider(s) "
            f"({', '.join(sorted(cloud_providers_selected))}). "
            "Make sure the matching API key(s) are set in .env "
            "before running the app."
        )

    print("\nSetup complete.")

def run_non_interactive():
    extras_map = _load_extras_map()
    current_values = _load_env_values()

    choices = {}
    for role, env_var in ROLE_ENV_VARS.items():
        provider = current_values.get(env_var)
        if not provider:
            print(
                f"ERROR: {env_var} is not set in .env, and "
                "--non-interactive mode cannot prompt for it.\n"
                f"Set {env_var} in .env (e.g. {env_var}=local) before "
                "running non-interactive setup, or run without "
                "--non-interactive for the interactive wizard."
            )
            sys.exit(1)
        choices[env_var] = provider

    extras = _extras_for_choices(extras_map, choices)
    success = _run_uv_sync(extras)

    if not success:
        print("\n`uv sync` failed -- see the error above.")
        sys.exit(1)

    print("Non-interactive setup complete.")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip prompts; read *_PROVIDER values already in .env and just run uv sync.",
    )
    args = parser.parse_args()

    non_interactive = args.non_interactive or os.getenv(
        "SETUP_NON_INTERACTIVE", ""
    ).lower() in ("true", "1", "yes")

    if non_interactive:
        run_non_interactive()
    else:
        run_interactive()

if __name__ == "__main__":
    main()