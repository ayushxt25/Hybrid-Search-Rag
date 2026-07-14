import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.api.dependencies import get_document_indexing_service  # noqa: E402

SUPPORTED_EXTENSIONS = {".md", ".txt"}
DEFAULT_CORPUS_DIR = Path("datasets/evaluation/corpus")
DEFAULT_OUTPUT = Path("reports/evaluation_corpus_manifest.json")


def discover_corpus_files(corpus_dir: Path) -> list[Path]:
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory does not exist: {corpus_dir}")

    visible_files = sorted(
        (
            path
            for path in corpus_dir.iterdir()
            if path.is_file() and not path.name.startswith(".")
        ),
        key=lambda path: path.name,
    )
    unsupported_files = [
        path
        for path in visible_files
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS
    ]

    if unsupported_files:
        names = ", ".join(path.name for path in unsupported_files)
        raise ValueError(f"Unsupported corpus file(s): {names}")

    if not visible_files:
        raise ValueError(
            f"Corpus directory contains no .md or .txt files: {corpus_dir}"
        )

    return visible_files


def build_document_manifest(ingested_document) -> dict[str, Any]:
    document = ingested_document.document

    return {
        "file_name": document.file_name,
        "document_id": document.document_id,
        "content_hash": document.content_hash,
        "chunk_count": ingested_document.chunk_count,
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "section_index": chunk.section_index,
                "heading": chunk.heading,
                "page_number": chunk.page_number,
                "text": chunk.text,
            }
            for chunk in ingested_document.chunks
        ],
    }


def prepare_corpus_manifest(
    *,
    corpus_dir: Path,
    output: Path,
    force: bool = False,
    indexing_service=None,
) -> dict[str, Any]:
    if output.exists() and not force:
        raise FileExistsError(
            f"Output already exists. Use --force to replace: {output}"
        )

    corpus_files = discover_corpus_files(corpus_dir)
    service = (
        indexing_service
        if indexing_service is not None
        else get_document_indexing_service()
    )

    documents = []

    for corpus_file in corpus_files:
        corpus_file.read_bytes()
        ingested_document = service.index_document_for_internal_use(corpus_file)
        documents.append(build_document_manifest(ingested_document))

    manifest = {
        "corpus_directory": corpus_dir.as_posix(),
        "document_count": len(documents),
        "chunk_count": sum(document["chunk_count"] for document in documents),
        "documents": documents,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Index the controlled evaluation corpus and write a chunk manifest."
        ),
    )
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        manifest = prepare_corpus_manifest(
            corpus_dir=args.corpus_dir,
            output=args.output,
            force=args.force,
        )
        print(
            f"Indexed {manifest['document_count']} documents and "
            f"{manifest['chunk_count']} chunks."
        )
        print(f"Manifest written to: {args.output}")
        return 0
    except Exception as error:
        print(f"Evaluation corpus preparation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
