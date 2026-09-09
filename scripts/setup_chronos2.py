"""Download only the pinned public checkpoint; no fund data or credentials are sent."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import httpx

REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
WEIGHT_HASH = "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42"
ROOT = Path(__file__).resolve().parents[1]


def download_weights(client, destination):
    """Bounded ranged requests resume validated chunks after an interrupted large CDN response."""
    size, chunk_size = 477930472, 8 * 1024 * 1024
    chunks = destination / "download-chunks"
    chunks.mkdir(exist_ok=True)

    def fetch(start):
        end = min(size, start + chunk_size) - 1
        path = chunks / f"{start}.bin"
        if path.exists() and path.stat().st_size == end - start + 1:
            return path
        for attempt in range(4):
            try:
                response = client.get(
                    f"https://huggingface.co/amazon/chronos-2/resolve/{REVISION}/model.safetensors?download=true",
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=60,
                )
                response.raise_for_status()
                if (
                    response.status_code != 206
                    or response.headers.get("content-range") != f"bytes {start}-{end}/{size}"
                ):
                    raise ValueError("unexpected checkpoint byte range")
                if len(response.content) != end - start + 1:
                    raise ValueError("incomplete checkpoint chunk")
                path.write_bytes(response.content)
                print(f"download chunk {start // chunk_size + 1}/57", flush=True)
                return path
            except (httpx.HTTPError, ValueError):
                if attempt == 3:
                    raise
        raise RuntimeError("unreachable")

    with ThreadPoolExecutor(max_workers=4) as executor:
        paths = list(executor.map(fetch, range(0, size, chunk_size)))
    partial = destination / "model.safetensors.part"
    with partial.open("wb") as stream:
        for path in paths:
            stream.write(path.read_bytes())
    partial.rename(destination / "model.safetensors")


def main():
    destination = ROOT / ".local-runs" / "model-cache" / REVISION
    destination.mkdir(parents=True, exist_ok=True)
    files, started = {}, perf_counter()
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for name in ("config.json", "model.safetensors", "README.md"):
            path = destination / name
            if name == "model.safetensors" and not path.exists():
                download_weights(client, destination)
            if not path.exists():
                temporary = destination / f"{name}.part"
                # Restart only this incomplete download; completed files are hash checked below.
                with client.stream(
                    "GET", f"https://huggingface.co/amazon/chronos-2/resolve/{REVISION}/{name}"
                ) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as stream:
                        for chunk in response.iter_bytes(1024 * 1024):
                            stream.write(chunk)
                temporary.rename(path)
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            files[name] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
            if name == "model.safetensors" and files[name]["sha256"] != WEIGHT_HASH:
                raise ValueError("official weight SHA256 mismatch")
            print(json.dumps({name: files[name]}), flush=True)
    manifest = {"revision": REVISION, "files": files, "download_seconds": perf_counter() - started}
    manifest_path = destination / "download.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
