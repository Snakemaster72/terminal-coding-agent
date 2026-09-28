from pathlib import Path


def resolve_path(base: str | Path, path: str | Path):
    path = Path(path)
    if path.is_absolute():
        return path.resolve()

    return (Path(base).resolve() / path).resolve()


def workspace_violation(
    path: str | Path, workspace: str | Path, jail: bool = True
) -> str | None:
    """Return an error message if `path` escapes `workspace`, else None.

    Both sides are fully resolved first, so `../` traversal and symlinks that
    point outside the workspace are both caught.
    """
    if not jail:
        return None

    root = Path(workspace).resolve()
    target = Path(path).resolve()

    if target == root or target.is_relative_to(root):
        return None

    return (
        f"Path '{target}' is outside the workspace '{root}'. "
        "Access outside the working directory is disabled; "
        "set workspace_jail = false in config.toml to allow it."
    )


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
