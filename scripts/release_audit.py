"""Fail when release artifacts that should stay outside Git are present."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_SIZE = 50 * 1024 * 1024
FORBIDDEN_SUFFIXES = {".bin", ".ckpt", ".dll", ".o", ".obj", ".pth", ".pyd", ".so"}
IGNORED_PARTS = {".git", "__pycache__"}


def main() -> int:
    issues: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in IGNORED_PARTS for part in path.parts):
            continue
        relative = path.relative_to(ROOT)
        if path.stat().st_size > MAX_FILE_SIZE:
            issues.append(f"large file: {relative} ({path.stat().st_size} bytes)")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            issues.append(f"generated/binary artifact: {relative}")

    if issues:
        print("Release audit failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("Release audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
