from __future__ import annotations

import ast
import json
import os
import shutil
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:i]
    return line


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except Exception:
            inner = value[1:-1].strip()
            if not inner:
                return []
            return [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _fallback_yaml_load(text: str) -> Any:
    raw_lines = []
    for raw in text.splitlines():
        stripped = _strip_comment(raw).rstrip()
        if stripped.strip():
            raw_lines.append(stripped)

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(raw_lines):
            return {}, index
        current = raw_lines[index]
        current_indent = len(current) - len(current.lstrip(" "))
        if current_indent < indent:
            return {}, index
        if current.lstrip().startswith("- "):
            items: list[Any] = []
            while index < len(raw_lines):
                line = raw_lines[index]
                line_indent = len(line) - len(line.lstrip(" "))
                content = line.strip()
                if line_indent != indent or not content.startswith("- "):
                    break
                rest = content[2:].strip()
                index += 1
                if rest == "":
                    nested, index = parse_block(index, indent + 2)
                    items.append(nested)
                elif ":" in rest and not rest.startswith(("http://", "https://")):
                    key, value = rest.split(":", 1)
                    obj: dict[str, Any] = {key.strip(): _parse_scalar(value)}
                    if index < len(raw_lines):
                        nxt_indent = len(raw_lines[index]) - len(raw_lines[index].lstrip(" "))
                        if nxt_indent > indent:
                            nested, index = parse_block(index, indent + 2)
                            if isinstance(nested, dict):
                                obj.update(nested)
                    items.append(obj)
                else:
                    items.append(_parse_scalar(rest))
            return items, index

        mapping: dict[str, Any] = {}
        while index < len(raw_lines):
            line = raw_lines[index]
            line_indent = len(line) - len(line.lstrip(" "))
            if line_indent < indent:
                break
            if line_indent > indent:
                nested, index = parse_block(index, line_indent)
                if isinstance(nested, dict):
                    mapping.update(nested)
                continue
            content = line.strip()
            if content.startswith("- "):
                break
            if ":" not in content:
                raise ValueError(f"Cannot parse YAML line: {line}")
            key, value = content.split(":", 1)
            key = key.strip()
            value = value.strip()
            index += 1
            if value == "":
                nested, index = parse_block(index, indent + 2)
                mapping[key] = nested
            else:
                mapping[key] = _parse_scalar(value)
        return mapping, index

    parsed, _ = parse_block(0, 0)
    return parsed


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text()
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except Exception:
        data = _fallback_yaml_load(text)
    return data or {}


def dump_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    try:
        import yaml  # type: ignore

        path.write_text(yaml.safe_dump(data, sort_keys=False))
    except Exception:
        path.write_text(json.dumps(data, indent=2, sort_keys=False))


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=False, default=str))


def copy_if_exists(src: str | Path, dst: str | Path) -> bool:
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        return False
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return True


def project_path(root: str | Path, maybe_relative: str | Path) -> Path:
    p = Path(maybe_relative)
    return p if p.is_absolute() else Path(root) / p


def atomic_stage_marker(run_dir: str | Path, stage: str) -> Path:
    return Path(run_dir) / "logs" / f"{stage}.done"
