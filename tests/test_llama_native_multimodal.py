import ctypes
import gc
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

import llama_cpp.llama_multimodal as multimodal_module
from llama_cpp.llama_multimodal import MTMDChatHandler


class _Chunk:
    def __init__(self, chunk_type, *, tokens=None, n_tokens=0):
        self.chunk_type = chunk_type
        self.tokens = list(tokens or [])
        self.n_tokens = n_tokens or len(self.tokens)
        self.token_array = (ctypes.c_int32 * len(self.tokens))(*self.tokens)


class _FakeMTMD:
    mtmd_input_chunk_type = SimpleNamespace(
        MTMD_INPUT_CHUNK_TYPE_TEXT=0,
        MTMD_INPUT_CHUNK_TYPE_IMAGE=1,
        MTMD_INPUT_CHUNK_TYPE_AUDIO=2,
    )
    mtmd_helper_post_decode_callback = staticmethod(lambda callback: callback)

    def __init__(self, events, *, failure=None):
        self.events = events
        self.failure = failure
        self.callback_ref = None

    def mtmd_input_chunk_get_type(self, chunk):
        return chunk.chunk_type

    def mtmd_input_chunk_get_n_tokens(self, chunk):
        return chunk.n_tokens

    def mtmd_input_chunk_get_tokens_text(self, chunk, n_tokens_out):
        ctypes.cast(n_tokens_out, ctypes.POINTER(ctypes.c_size_t))[0] = len(
            chunk.tokens
        )
        return chunk.token_array

    def mtmd_batch_init(self, context):
        self.events.append(("batch_init", context))
        return object()

    def mtmd_batch_add_chunk(self, batch, chunk):
        self.events.append(("add_chunk", chunk))
        return 2 if self.failure == "add" else 0

    def mtmd_batch_encode(self, batch):
        self.events.append(("encode",))
        return 1 if self.failure == "encode" else 0

    def mtmd_batch_get_output_embd(self, batch, chunk):
        self.events.append(("get_embd",))
        if self.failure == "embd":
            return None
        return object()

    def mtmd_helper_decode_image_chunk(
        self,
        context,
        target_context,
        chunk,
        encoded_embd,
        n_past,
        seq_id,
        n_batch,
        new_n_past,
        callback,
        user_data,
    ):
        del context, target_context, encoded_embd, seq_id
        self.events.append(("target_decode_start",))
        if self.failure == "decode":
            return 7

        self.callback_ref = weakref.ref(callback)
        gc.collect()
        assert self.callback_ref() is callback

        remaining = chunk.n_tokens
        while remaining:
            size = min(remaining, n_batch)
            batch = SimpleNamespace(
                n_tokens=size,
                token=None,
                embd=object(),
            )
            self.events.append(("target_decode", size))
            result = callback(batch, user_data)
            if result != 0:
                return result
            remaining -= size

        start = n_past.value if hasattr(n_past, "value") else int(n_past)
        ctypes.cast(new_n_past, ctypes.POINTER(ctypes.c_int32))[0] = (
            start + chunk.n_tokens
        )
        return 0

    def mtmd_batch_free(self, batch):
        self.events.append(("batch_free", batch))

    def mtmd_input_chunks_free(self, chunks):
        self.events.append(("chunks_free", chunks))

    def mtmd_bitmap_free(self, bitmap):
        self.events.append(("bitmap_free", bitmap))


class _FakeContext:
    def __init__(self, events):
        self.events = events
        self.ctx = object()
        self.clear_count = 0

    def memory_clear(self, clear_data):
        self.clear_count += 1
        self.events.append(("target_clear", bool(clear_data)))


class _FakeNative:
    def __init__(self, events, *, process_error=None):
        self.events = events
        self.process_error = process_error
        self.clear_count = 0
        self.media_batches = []
        self.text_batches = []

    def clear(self):
        self.clear_count += 1
        self.events.append(("native_clear",))

    def _process_batch(self, batch):
        if batch.embd is not None:
            assert batch.token is None
            self.media_batches.append(batch.n_tokens)
            self.events.append(("native_media_process", batch.n_tokens))
            if self.process_error is not None:
                raise self.process_error
        else:
            self.text_batches.append(batch.n_tokens)
            self.events.append(("native_text_process", batch.n_tokens))


class _FakeLlama:
    def __init__(self, events, native):
        self.events = events
        self._native_speculative = native
        self._ctx = _FakeContext(events)
        self._hybrid_cache_mgr = None
        self.is_hybrid = False
        self.verbose = False
        self.n_batch = 2
        self.n_tokens = 3
        self.input_ids = np.full(32, 99, dtype=np.intc)
        self.completion_prompt = None

    def n_ctx(self):
        return len(self.input_ids)

    def longest_token_prefix(self, *args, **kwargs):
        raise AssertionError("native multimodal prefill must not reuse a prefix")

    def eval(self, tokens):
        tokens = list(tokens)
        start = self.n_tokens
        self.input_ids[start : start + len(tokens)] = tokens
        self.n_tokens += len(tokens)
        batch = SimpleNamespace(
            n_tokens=len(tokens),
            token=object(),
            embd=None,
        )
        self._native_speculative._process_batch(batch)

    def create_completion(self, *, prompt, **kwargs):
        del kwargs
        self.completion_prompt = list(prompt)
        return {"ok": True}


def _make_handler(monkeypatch, *, failure=None, process_error=None):
    events = []
    handler = object.__new__(MTMDChatHandler)
    handler.verbose = False
    handler.log_prefix = "MTMD"
    handler.mtmd_ctx = object()
    handler._mtmd_cpp = _FakeMTMD(events, failure=failure)
    handler._init_mtmd_context = lambda llama: None

    text_before = _Chunk(0, tokens=[10, 11])
    image = _Chunk(1, n_tokens=5)
    text_after = _Chunk(0, tokens=[20])
    spans = [
        (0, 2, text_before, 0, None),
        (2, 7, image, 1, -123),
        (7, 8, text_after, 0, None),
    ]
    full_prompt = [10, 11, -123, -123, -123, -123, -123, 20]
    chunks = object()
    handler._process_mtmd_prompt = lambda **kwargs: (
        full_prompt,
        spans,
        chunks,
        [],
    )

    native = _FakeNative(events, process_error=process_error)
    llama = _FakeLlama(events, native)
    monkeypatch.setattr(
        multimodal_module,
        "_convert_completion_to_chat",
        lambda completion, stream: completion,
    )
    return handler, llama, native, events


def test_native_multimodal_prefill_processes_each_decode_batch_once(monkeypatch):
    handler, llama, native, events = _make_handler(monkeypatch)

    response = handler(llama=llama, messages=[])

    assert response == {"ok": True}
    assert llama._ctx.clear_count == 1
    assert native.clear_count == 1
    assert native.text_batches == [2, 1]
    assert native.media_batches == [2, 2, 1]
    assert llama.completion_prompt == [
        10,
        11,
        -123,
        -123,
        -123,
        -123,
        -123,
        20,
    ]
    assert llama.n_tokens == 8
    assert llama._native_has_media_context is True

    ordered_names = [event[0] for event in events]
    assert ordered_names == [
        "target_clear",
        "native_clear",
        "native_text_process",
        "batch_init",
        "add_chunk",
        "encode",
        "get_embd",
        "target_decode_start",
        "target_decode",
        "native_media_process",
        "target_decode",
        "native_media_process",
        "target_decode",
        "native_media_process",
        "batch_free",
        "native_text_process",
        "chunks_free",
    ]


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("add", "batch add"),
        ("encode", "media encode"),
        ("embd", "no output embeddings"),
        ("decode", "target media decode"),
    ],
)
def test_native_media_failures_free_batch_and_leave_ledger_uncommitted(
    monkeypatch, failure, message
):
    handler, llama, native, events = _make_handler(
        monkeypatch, failure=failure
    )

    with pytest.raises((RuntimeError, ValueError), match=message):
        handler(llama=llama, messages=[])

    assert sum(event[0] == "batch_free" for event in events) == 1
    assert llama.n_tokens == 0
    assert llama._ctx.clear_count == 2
    assert native.clear_count == 2
    assert not np.any(llama.input_ids[:5] == -123)


def test_native_callback_reraises_original_error_and_clears_both_contexts(
    monkeypatch,
):
    original = LookupError("draft bridge rejected embedding batch")
    handler, llama, native, events = _make_handler(
        monkeypatch, process_error=original
    )

    with pytest.raises(LookupError, match="draft bridge rejected") as exc_info:
        handler(llama=llama, messages=[])

    assert exc_info.value is original
    assert sum(event[0] == "batch_free" for event in events) == 1
    assert llama._ctx.clear_count == 2
    assert native.clear_count == 2
    assert llama.n_tokens == 0


def test_native_multimodal_rejects_audio_before_prefill(monkeypatch):
    handler, llama, native, events = _make_handler(monkeypatch)
    audio = _Chunk(2, n_tokens=3)
    handler._process_mtmd_prompt = lambda **kwargs: (
        [-456, -456, -456],
        [(0, 3, audio, 2, -456)],
        object(),
        [],
    )

    with pytest.raises(ValueError, match="image input only"):
        handler(llama=llama, messages=[])

    assert llama.n_tokens == 0
    assert llama._ctx.clear_count == 1
    assert native.clear_count == 1
    assert not any(event[0] == "batch_init" for event in events)


def test_native_multimodal_rejects_context_shift(monkeypatch):
    handler, llama, native, events = _make_handler(monkeypatch)
    llama.input_ids = np.full(7, 99, dtype=np.intc)

    with pytest.raises(RuntimeError, match="does not support context shifting"):
        handler(llama=llama, messages=[])

    assert llama.n_tokens == 0
    assert llama._ctx.clear_count == 2
    assert native.clear_count == 2
    assert not any(event[0] == "batch_init" for event in events)


def test_muse_patch_placeholder_is_normalized_for_mtmd():
    handler = object.__new__(MTMDChatHandler)
    handler.log_prefix = "MTMD"
    handler.media_marker = "<__media__>"
    handler._chat_format_parser_tags = ["<|patch|>"]

    rendered = handler._replace_media_placeholders(
        "before<|patch|>after",
        [{"type": "image", "url": "image.jpg"}],
    )

    assert rendered == "before<__media__>after"
