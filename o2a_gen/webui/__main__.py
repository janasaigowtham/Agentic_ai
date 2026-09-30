"""python -m o2a_gen.webui [--host 127.0.0.1] [--port 8765]"""

import argparse

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser(prog="o2a_gen.webui")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    print(f"o2a_gen UI: http://{args.host}:{args.port}")
    uvicorn.run("o2a_gen.webui.server:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
