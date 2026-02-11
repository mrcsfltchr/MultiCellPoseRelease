import sys

from guv_app.plugins.plugin_validator import validate_all_plugins


def main() -> int:
    results = validate_all_plugins()
    if not results:
        print("No plugins discovered.")
        return 1

    any_error = False
    for r in results:
        status = "PASS" if r.valid else "FAIL"
        print(f"[{status}] {r.plugin_name} ({r.class_name}) in {r.module}")
        for w in r.warnings:
            print(f"  WARN: {w}")
        for e in r.errors:
            print(f"  ERROR: {e}")
        if r.errors:
            any_error = True

    return 1 if any_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
