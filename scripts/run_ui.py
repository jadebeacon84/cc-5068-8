"""Start the local web interface.

    python3 -m scripts.run_ui [--port 5000] [--host 127.0.0.1]

Then open the printed URL in a browser. The server only serves local files and
kernel results; it never calls a remote recognition API.
"""
from __future__ import annotations

import argparse

from ser.webapp import app, pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    ready = (pipeline.data_root / "manifest.json").exists()
    if not ready:
        print("提示：数据集尚未生成，请先依次运行 "
              "python3 -m scripts.prepare_data 和 python3 -m scripts.train_model")
    print(f"界面地址: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
