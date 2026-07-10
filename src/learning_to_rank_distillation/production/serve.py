"""Uvicorn entrypoint for the ranking service."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from learning_to_rank_distillation.production.serving import create_app

app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a student ranking bundle.")
    parser.add_argument("--bundle-path", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    target = "learning_to_rank_distillation.production.serve:app"
    if args.bundle_path is not None:
        app_with_bundle = create_app(bundle_path=args.bundle_path)
        uvicorn.run(app_with_bundle, host=args.host, port=args.port)
    else:
        uvicorn.run(target, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
