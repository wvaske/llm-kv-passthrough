"""Load pinned SGLang storage code without importing its GPU runtime.

Storage method bodies are upstream code, not a reimplementation. Only imports
of four SGLang dependencies are redirected to isolated, verified source modules.
No fake sglang modules are installed and no global Torch settings are changed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import re
import sys
import types
from decimal import Decimal
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any

VERSION = "0.5.21"
REVISION = "e00930c5489053f26d86b179cee0d087f846acbb"
SOURCE_HASHES = {
    "srt/environ.py": "2085881c2aa46abd90fc43af6b8f4672a0e3377f18eb2393d165a7a5d9e98a6d",
    "srt/utils/common.py": "71abb02aca154dfd2f30ed7e13b6550ac98ec9883b32e73e744e317552730fea",
    "srt/mem_cache/hicache_storage.py": "f54c2e54fd2e7546a0814dba922f6264362049fcbf1e390dde738144838f5733",
    "srt/mem_cache/storage/file/lru_file_evictor.py": "9e590e144c1169d463102e9aa147432cf5fa9108f1f8936de1184903c79a1330",
}


class _StorageImports(ast.NodeTransformer):
    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        if node.module in {
            "sglang.srt.environ", "sglang.srt.utils.common",
            "sglang.srt.mem_cache.storage.file.lru_file_evictor",
            "sglang.srt.mem_cache.pool_host",
        }:
            return ast.copy_location(ast.Pass(), node)
        return node


@lru_cache(maxsize=1)
def load_hicache() -> types.ModuleType:
    try:
        package = distribution("sglang")
    except PackageNotFoundError as exc:
        raise RuntimeError("HiCache requires `uv pip install --no-deps sglang==0.5.21` and CPU Torch") from exc
    if package.version != VERSION:
        raise RuntimeError(f"HiCache requires SGLang {VERSION}, found {package.version}")

    def source(relative: str) -> ast.Module:
        path = Path(str(package.locate_file("sglang/" + relative)))
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != SOURCE_HASHES[relative]:
            raise RuntimeError(f"Unsupported/modified SGLang storage source: {relative}")
        return ast.parse(raw, filename=str(path))

    def execute(name: str, tree: ast.Module, **dependencies: Any) -> types.ModuleType:
        module = types.ModuleType(f"{__name__}._{name}")
        module.__dict__.update(dependencies)
        sys.modules[module.__name__] = module  # dataclasses resolves __module__
        try:
            exec(compile(ast.fix_missing_locations(tree), module.__name__, "exec"), module.__dict__)
        except BaseException:
            sys.modules.pop(module.__name__, None)
            raise
        return module

    env = execute("environ", source("srt/environ.py"))
    common = source("srt/utils/common.py")
    size_parser = ast.Module(body=[n for n in common.body if isinstance(n, ast.FunctionDef)
                                  and n.name == "human_readable_int"], type_ignores=[])
    if len(size_parser.body) != 1:
        raise RuntimeError("SGLang size parser changed")
    helper = execute("size", size_parser, re=re, argparse=argparse, Decimal=Decimal)
    evictor = execute("evictor", _StorageImports().visit(source("srt/mem_cache/storage/file/lru_file_evictor.py")),
                      envs=env.envs, human_readable_int=helper.human_readable_int)
    return execute("storage", _StorageImports().visit(source("srt/mem_cache/hicache_storage.py")),
                   envs=env.envs, LRUFileEvictor=evictor.LRUFileEvictor)
