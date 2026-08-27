from pathlib import Path


def resolve_path(base: str | Path, path: str | Path):
    path = Path(path)
    if path.is_absolute():
        return path.resolve()

    return (Path(base).resolve() / path).resolve()


def resolve_path_rel_to_cwd(path: str, cwd: Path) -> str:
    try:
        p = Path(path)
    except Exception:
        return str(path)

    if cwd:
        try:
            return str(p.relative_to(cwd))
        except ValueError:
            pass
    return str(p)


def ensure_parent_dir(path: str | Path) -> Path:
    path = Path(path)
    parent_dir = path.parent
    if not parent_dir.exists():
        parent_dir.mkdir(parents=True, exist_ok=True)

    return path


def is_binary_file(path: str | Path) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
            return b"\x00" in chunk

    except OSError:
        return False
