from __future__ import annotations

from typing import Optional

import pytest

from llama_cpp import Llama


class _PositionTrackingContext:
    """Small target-context stand-in that detects stale position reuse."""

    def __init__(self) -> None:
        self.positions: set[int] = set()
        self.clear_calls: list[bool] = []

    def decode(self, position: int, count: int) -> None:
        requested = set(range(position, position + count))
        if requested & self.positions:
            raise RuntimeError("Invalid input batch")
        self.positions.update(requested)

    def memory_clear(self, clear_data: bool) -> None:
        self.clear_calls.append(clear_data)
        self.positions.clear()


class _CheckpointManager:
    def __init__(self) -> None:
        self.checkpoints: list[int] = []
        self.clear_count = 0

    def clear(self) -> None:
        self.clear_count += 1
        self.checkpoints.clear()


class _NativeDecoder:
    def __init__(self) -> None:
        self.clear_count = 0

    def clear(self) -> None:
        self.clear_count += 1


class _ModelLifecycle:
    def __init__(self) -> None:
        self.create_count = 0
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


def _new_llama(
    *,
    lifecycle: Optional[_ModelLifecycle] = None,
    checkpoint_manager: Optional[_CheckpointManager] = None,
    native_decoder: Optional[_NativeDecoder] = None,
) -> Llama:
    if lifecycle is not None:
        lifecycle.create_count += 1
    llama = Llama.__new__(Llama)
    llama._ctx = _PositionTrackingContext()
    llama._hybrid_cache_mgr = checkpoint_manager
    llama._native_speculative = native_decoder
    llama._native_has_media_context = False
    llama.n_tokens = 0
    llama._stack = lifecycle
    return llama


def _run_request(llama: Llama, tokens: list[int], *, has_media: bool = False) -> None:
    # Model the invariant enforced by llama_decode: a fresh request starts at
    # the Python cursor, and position 0 must not already exist in target memory.
    llama._ctx.decode(llama.n_tokens, len(tokens))
    llama.n_tokens += len(tokens)
    llama._native_has_media_context = has_media

    checkpoint_manager = llama._hybrid_cache_mgr
    if checkpoint_manager is not None:
        checkpoint_manager.checkpoints.append(llama.n_tokens)


@pytest.mark.parametrize(
    ("request_kind", "use_checkpoints", "use_native"),
    [
        ("transformer", False, False),
        ("hybrid_swa", True, False),
        ("image", False, False),
        ("audio", False, False),
        ("native_speculative", False, True),
    ],
)
def test_reset_allows_independent_requests_from_position_zero(
    request_kind: str, use_checkpoints: bool, use_native: bool
) -> None:
    """One model/context serves two independent requests with one reset."""
    lifecycle = _ModelLifecycle()
    checkpoint_manager = _CheckpointManager() if use_checkpoints else None
    native_decoder = _NativeDecoder() if use_native else None

    llama = _new_llama(
        lifecycle=lifecycle,
        checkpoint_manager=checkpoint_manager,
        native_decoder=native_decoder,
    )
    try:
        has_media = request_kind in ("image", "audio")
        _run_request(llama, [10, 11, 12], has_media=has_media)

        llama.reset()

        assert llama.n_tokens == 0
        assert llama._native_has_media_context is False
        _run_request(llama, [20, 21], has_media=has_media)
        assert llama.n_tokens == 2
        assert llama._ctx.positions == {0, 1}
        assert llama._ctx.clear_calls == [True]
        if checkpoint_manager is not None:
            assert checkpoint_manager.clear_count == 1
            assert checkpoint_manager.checkpoints == [2]
        if native_decoder is not None:
            assert native_decoder.clear_count == 1
    finally:
        llama.close()

    assert lifecycle.create_count == 1
    assert lifecycle.close_count == 1


def test_reset_clears_hybrid_swa_memory_and_checkpoints() -> None:
    checkpoint_manager = _CheckpointManager()
    llama = _new_llama(checkpoint_manager=checkpoint_manager)

    _run_request(llama, [1, 2, 3])
    assert checkpoint_manager.checkpoints == [3]

    llama.reset()

    assert llama.n_tokens == 0
    assert llama._ctx.clear_calls == [True]
    assert checkpoint_manager.clear_count == 1
    assert checkpoint_manager.checkpoints == []
    _run_request(llama, [4])
    assert llama._ctx.positions == {0}


def test_reset_keeps_native_speculative_contexts_in_sync() -> None:
    native_decoder = _NativeDecoder()
    llama = _new_llama(native_decoder=native_decoder)
    _run_request(llama, [1, 2], has_media=True)

    llama.reset()

    assert llama._ctx.clear_calls == [True]
    assert native_decoder.clear_count == 1
    assert llama._native_has_media_context is False
    assert llama.n_tokens == 0
