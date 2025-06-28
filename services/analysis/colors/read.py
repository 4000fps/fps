import gzip
import json
import pprint
import sys


def read(file_path: str) -> list[dict]:
    """Read a JSONL file and return a list of dictionaries."""
    with gzip.open(file_path, "rt") as file:
        return [json.loads(line) for line in file]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    data = read(file_path)
    pprint.pprint(data)
