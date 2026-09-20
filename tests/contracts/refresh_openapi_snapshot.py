"""Rewrite openapi_snapshot.json from the live app. Run only when an API change
is intentional; commit the result in the same PR. See test_openapi_snapshot.py."""

import json
from pathlib import Path

from ._api_contract import current_projection

SNAPSHOT = Path(__file__).parent / "openapi_snapshot.json"


def main() -> None:
    SNAPSHOT.write_text(json.dumps(current_projection(), indent=2, sort_keys=True) + "\n")
    print(f"wrote {SNAPSHOT}")


if __name__ == "__main__":
    main()
