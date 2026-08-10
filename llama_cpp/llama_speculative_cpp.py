"""Experimental ctypes bindings for llama.cpp/common speculative decoding.

This module is intentionally loaded lazily by ``LlamaNativeSpeculativeDecoding``.
The bridge lives in ``llama-common`` rather than the stable public ``llama`` C API.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional

from llama_cpp import llama_cpp
from llama_cpp._ctypes_extensions import (
    ctypes_function_for_shared_library,
    load_shared_library,
)


LLAMA_CPP_NATIVE_SPECULATIVE_ABI_VERSION = 1


def _load_llama_common() -> ctypes.CDLL:
    """Prefer the wheel-local library over a potentially stale system copy."""
    override = os.environ.get("LLAMA_CPP_LIB_PATH")
    lib_dir = (
        Path(override)
        if override is not None
        else Path(__file__).resolve().parent / "lib"
    )
    if sys.platform == "win32":
        names = ("llama-common.dll", "libllama-common.dll")
        kwargs = {"winmode": ctypes.RTLD_GLOBAL}
    elif sys.platform == "darwin":
        names = ("libllama-common.dylib", "libllama-common.so")
        kwargs = {}
    else:
        names = ("libllama-common.so",)
        kwargs = {}

    for name in names:
        path = lib_dir / name
        if path.exists():
            return ctypes.CDLL(str(path), **kwargs)
    return load_shared_library("llama-common", [lib_dir])


_lib = _load_llama_common()
ctypes_function = ctypes_function_for_shared_library(_lib)


class llama_cpp_native_speculative(ctypes.Structure):
    pass


llama_cpp_native_speculative_p = ctypes.POINTER(llama_cpp_native_speculative)


class llama_cpp_native_speculative_params(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("model_path", ctypes.c_char_p),
        ("spec_type", ctypes.c_char_p),
        ("n_gpu_layers", ctypes.c_int32),
        ("n_ctx", ctypes.c_int32),
        ("n_batch", ctypes.c_int32),
        ("n_ubatch", ctypes.c_int32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("n_max", ctypes.c_int32),
        ("n_min", ctypes.c_int32),
        ("p_min", ctypes.c_float),
        ("cache_type_k", ctypes.c_int32),
        ("cache_type_v", ctypes.c_int32),
        ("flash_attn_type", ctypes.c_int32),
        ("offload_kqv", ctypes.c_bool),
        ("op_offload", ctypes.c_bool),
        ("kv_unified", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
    ]


@ctypes_function(
    "llama_cpp_native_speculative_abi_version",
    [],
    ctypes.c_uint32,
)
def llama_cpp_native_speculative_abi_version() -> int:
    ...


if llama_cpp_native_speculative_abi_version() != LLAMA_CPP_NATIVE_SPECULATIVE_ABI_VERSION:
    raise RuntimeError(
        "llama-common native speculative ABI mismatch; rebuild/reinstall "
        "llama-cpp-python from this experimental branch"
    )


@ctypes_function(
    "llama_cpp_native_speculative_init",
    [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(llama_cpp_native_speculative_params),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ],
    llama_cpp_native_speculative_p,
)
def llama_cpp_native_speculative_init(
    target_model: ctypes.c_void_p,
    target_context: ctypes.c_void_p,
    params: ctypes.POINTER(llama_cpp_native_speculative_params),
    error: ctypes.POINTER(ctypes.c_char),
    error_capacity: int,
) -> Optional[llama_cpp_native_speculative_p]:
    ...


@ctypes_function(
    "llama_cpp_native_speculative_free",
    [llama_cpp_native_speculative_p],
    None,
)
def llama_cpp_native_speculative_free(
    speculative: llama_cpp_native_speculative_p,
) -> None:
    ...


@ctypes_function(
    "llama_cpp_native_speculative_last_error",
    [llama_cpp_native_speculative_p],
    ctypes.c_char_p,
)
def llama_cpp_native_speculative_last_error(
    speculative: llama_cpp_native_speculative_p,
) -> bytes:
    ...


@ctypes_function(
    "llama_cpp_native_speculative_begin",
    [
        llama_cpp_native_speculative_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_size_t,
    ],
    ctypes.c_bool,
)
def llama_cpp_native_speculative_begin(
    speculative: llama_cpp_native_speculative_p,
    prompt_tokens: ctypes.POINTER(ctypes.c_int32),
    prompt_token_count: int,
) -> bool:
    ...


@ctypes_function(
    "llama_cpp_native_speculative_process",
    [
        llama_cpp_native_speculative_p,
        ctypes.POINTER(llama_cpp.llama_batch),
    ],
    ctypes.c_bool,
)
def llama_cpp_native_speculative_process(
    speculative: llama_cpp_native_speculative_p,
    batch: ctypes.POINTER(llama_cpp.llama_batch),
) -> bool:
    ...


@ctypes_function(
    "llama_cpp_native_speculative_draft",
    [
        llama_cpp_native_speculative_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_size_t,
    ],
    ctypes.c_int32,
)
def llama_cpp_native_speculative_draft(
    speculative: llama_cpp_native_speculative_p,
    n_past: int,
    id_last: int,
    prompt_tokens: ctypes.POINTER(ctypes.c_int32),
    prompt_token_count: int,
    output_tokens: ctypes.POINTER(ctypes.c_int32),
    output_capacity: int,
) -> int:
    ...


@ctypes_function(
    "llama_cpp_native_speculative_accept",
    [llama_cpp_native_speculative_p, ctypes.c_uint16],
    ctypes.c_bool,
)
def llama_cpp_native_speculative_accept(
    speculative: llama_cpp_native_speculative_p,
    n_accepted: int,
) -> bool:
    ...


@ctypes_function(
    "llama_cpp_native_speculative_memory_seq_rm",
    [llama_cpp_native_speculative_p, ctypes.c_int32, ctypes.c_int32],
    ctypes.c_bool,
)
def llama_cpp_native_speculative_memory_seq_rm(
    speculative: llama_cpp_native_speculative_p,
    p0: int,
    p1: int,
) -> bool:
    ...


@ctypes_function(
    "llama_cpp_native_speculative_memory_seq_add",
    [
        llama_cpp_native_speculative_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ],
    ctypes.c_bool,
)
def llama_cpp_native_speculative_memory_seq_add(
    speculative: llama_cpp_native_speculative_p,
    p0: int,
    p1: int,
    delta: int,
) -> bool:
    ...


@ctypes_function(
    "llama_cpp_native_speculative_memory_clear",
    [llama_cpp_native_speculative_p, ctypes.c_bool],
    ctypes.c_bool,
)
def llama_cpp_native_speculative_memory_clear(
    speculative: llama_cpp_native_speculative_p,
    clear_data: bool,
) -> bool:
    ...


@ctypes_function(
    "llama_cpp_native_speculative_print_stats",
    [llama_cpp_native_speculative_p],
    None,
)
def llama_cpp_native_speculative_print_stats(
    speculative: llama_cpp_native_speculative_p,
) -> None:
    ...
