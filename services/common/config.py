from pathlib import Path
from typing import Any

import yaml


def load_config(file_path: str | Path) -> dict[str, Any]:
    # Load config file
    with open(file_path, "r") as yaml_file:
        config: dict[str, Any] = yaml.safe_load(yaml_file)

    # Fill default URL values
    def _fill_null(dictionary: dict[str, str], fill_value: str) -> None:
        for key, value in dictionary.items():
            if value is None:
                dictionary[key] = fill_value

    _fill_null(config["static_files_urls"], "http://router")
    _fill_null(config["services_urls"], "http://router")

    return config


if __name__ == "__main__":
    pass
