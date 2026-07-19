from __future__ import annotations

from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STANDARD_PYTHON_LIMIT_BYTES = 500 * 1024 * 1024
REQUIRED_RUNTIME_PATHS = [
    ROOT / "api" / "index.py",
    ROOT / "backend" / "app",
    ROOT / "requirements-vercel.txt",
]
FORBIDDEN_REQUIREMENTS = [
    "torch",
    "torchvision",
    "scipy",
    "transformers",
    "sentence-transformers",
    "tokenizers",
]
FORBIDDEN_MODEL_CACHE_PATHS = [
    ROOT / ".cache" / "huggingface",
    ROOT / "huggingface",
    ROOT / "backend" / ".cache" / "huggingface",
]


def bytes_under(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def distribution_size(name: str) -> tuple[str, int]:
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return name, 0

    total = 0
    for file in dist.files or []:
        located = Path(dist.locate_file(file))
        if located.is_file():
            total += located.stat().st_size
    return dist.metadata["Name"], total


def requirement_names() -> list[str]:
    names = []
    for line in (
        (ROOT / "requirements-vercel.txt").read_text(encoding="utf-8").splitlines()
    ):
        normalized = line.strip()
        if not normalized or normalized.startswith("#"):
            continue
        names.append(normalized.split("<", 1)[0].split(">", 1)[0].split("=", 1)[0])
    return names


def format_mb(size: int) -> str:
    return f"{size / (1024 * 1024):.1f} MB"


def main() -> int:
    missing = [path for path in REQUIRED_RUNTIME_PATHS if not path.exists()]
    committed_caches = [path for path in FORBIDDEN_MODEL_CACHE_PATHS if path.exists()]
    requirements = (ROOT / "requirements-vercel.txt").read_text(encoding="utf-8")
    forbidden_requirements = [
        package for package in FORBIDDEN_REQUIREMENTS if package in requirements
    ]

    source_size = bytes_under(ROOT / "api") + bytes_under(ROOT / "backend" / "app")
    package_sizes = sorted(
        (distribution_size(name) for name in requirement_names()),
        key=lambda item: item[1],
        reverse=True,
    )
    estimated_size = source_size + sum(size for _, size in package_sizes)

    print("Vercel Python bundle audit")
    print(f"Runtime source estimate: {format_mb(source_size)}")
    print(f"Selected installed-package estimate: {format_mb(estimated_size)}")
    print("Largest selected packages:")
    for name, size in package_sizes[:10]:
        print(f"- {name}: {format_mb(size)}")

    if estimated_size > STANDARD_PYTHON_LIMIT_BYTES:
        print(
            "WARNING: selected installed packages exceed Vercel's standard "
            "500 MB Python bundle limit. This is an estimate, not Vercel's bundler."
        )

    if missing:
        print("ERROR: required runtime paths are missing:")
        for path in missing:
            print(f"- {path.relative_to(ROOT)}")
    if committed_caches:
        print("ERROR: model cache paths are present in the repository tree:")
        for path in committed_caches:
            print(f"- {path.relative_to(ROOT)}")
    if forbidden_requirements:
        print("ERROR: forbidden local ML packages are listed for Vercel:")
        for package in forbidden_requirements:
            print(f"- {package}")

    return 1 if missing or committed_caches or forbidden_requirements else 0


if __name__ == "__main__":
    raise SystemExit(main())
