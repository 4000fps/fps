import gzip
import json
import sys


def get_json_by_id(file_path: str, target_id: str) -> None:
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get("_id") == target_id:
                    print(json.dumps(obj, ensure_ascii=False, indent=2))
                    return
            except json.JSONDecodeError:
                continue
    print(f"No entry found with _id: {target_id}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <jsonl_file_path> <_id>", file=sys.stderr)
        sys.exit(1)
    file_path = sys.argv[1]
    target_id = str(sys.argv[2])
    get_json_by_id(file_path, target_id)
