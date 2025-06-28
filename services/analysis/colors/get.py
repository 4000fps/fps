import gzip
import json
import logging
import sys

from ...common.utils import setup_logging

setup_logging()
logger = logging.getLogger("services.analysis.colors.get")


def get_json_by_id(file_path: str, target_id: str) -> None:
    with gzip.open(file_path, "rt", encoding="utf-8") as file:
        for line in file:
            try:
                obj = json.loads(line)
                if obj.get("_id") == target_id:
                    print(json.dumps(obj, ensure_ascii=False, indent=2))
                    return
            except json.JSONDecodeError:
                logger.error(
                    f"Failed to decode JSON line: {line.strip()}", exc_info=True
                )
                continue
    logger.error(f"No record found with _id: {target_id}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <jsonl_file_path> <_id>", file=sys.stderr)
        sys.exit(1)
    file_path = sys.argv[1]
    target_id = str(sys.argv[2])
    get_json_by_id(file_path, target_id)
