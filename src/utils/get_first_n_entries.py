import zstandard as zstd
import json
import io


def get_first_n_entries(zst_path: str, n: int = 100_000):
    dctx = zstd.ZstdDecompressor()
    count = 0
    results = []
    if n == -1:
        n = float('inf')
        
    with open(zst_path, 'rb') as f:
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            for line in text_stream:
                try:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if "fen" in entry and "eval" in entry and isinstance(entry["eval"], (int, float)):
                        results.append({
                            "fen": entry["fen"],
                            "eval": int(entry["eval"])
                        })
                        count += 1
                    if count >= n:
                        break
                except Exception:
                    continue
    return results