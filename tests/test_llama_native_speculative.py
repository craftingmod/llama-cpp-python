import ctypes
import threading
from types import SimpleNamespace
from types import MethodType

import numpy as np
import pytest

from llama_cpp.llama_speculative import LlamaNativeSpeculativeDecoding
from llama_cpp.llama import Llama
import llama_cpp.llama as llama_module


class _FakeNativeParams(ctypes.Structure):
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


class _FakeBatch(ctypes.Structure):
    _fields_ = [("n_tokens", ctypes.c_int32)]


class _FakeModel:
    def __init__(self, value=101):
        self.model = ctypes.c_void_p(value)


class _FakeContext:
    def __init__(self, value=202):
        self.ctx = ctypes.c_void_p(value)
        self.params = SimpleNamespace(n_threads=4, n_threads_batch=8)

    def n_ctx(self):
        return 4096

    def n_batch(self):
        return 128

    def n_ubatch(self):
        return 64


def _read_i32(pointer, count):
    if count == 0:
        return []
    values = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int32))
    return [int(values[i]) for i in range(count)]


class _FakeNativeModule:
    llama_cpp_native_speculative_params = _FakeNativeParams

    def __init__(self):
        self.handle = object()
        self.init_error = None
        self.runtime_error = b"fake native failure"
        self.draft_result = [41, 42]
        self.draft_status = None
        self.calls = []
        self.init_params = None

    def llama_cpp_native_speculative_init(
        self, target_model, target_context, params_pointer, error, error_capacity
    ):
        params = ctypes.cast(
            params_pointer, ctypes.POINTER(_FakeNativeParams)
        ).contents
        self.init_params = {
            name: getattr(params, name) for name, _ctype in params._fields_
        }
        self.calls.append(("init", target_model, target_context))

        if self.init_error is not None:
            message = self.init_error.encode("utf-8") + b"\0"
            ctypes.memmove(error, message, min(len(message), error_capacity))
            return None
        return self.handle

    def llama_cpp_native_speculative_free(self, handle):
        self.calls.append(("free", handle))

    def llama_cpp_native_speculative_last_error(self, handle):
        self.calls.append(("last_error", handle))
        return self.runtime_error

    def llama_cpp_native_speculative_begin(self, handle, prompt, prompt_count):
        self.calls.append(
            ("begin", handle, _read_i32(prompt, int(prompt_count)))
        )
        return True

    def llama_cpp_native_speculative_process(self, handle, batch):
        raw_batch = ctypes.cast(batch, ctypes.POINTER(_FakeBatch)).contents
        self.calls.append(("process", handle, int(raw_batch.n_tokens)))
        return True

    def llama_cpp_native_speculative_draft(
        self,
        handle,
        n_past,
        id_last,
        prompt,
        prompt_count,
        output,
        output_capacity,
    ):
        self.calls.append(
            (
                "draft",
                handle,
                int(n_past),
                int(id_last),
                _read_i32(prompt, int(prompt_count)),
                int(output_capacity),
            )
        )
        if self.draft_status is not None:
            return self.draft_status

        output_values = ctypes.cast(output, ctypes.POINTER(ctypes.c_int32))
        count = min(len(self.draft_result), int(output_capacity))
        for i, token in enumerate(self.draft_result[:count]):
            output_values[i] = token
        return count

    def llama_cpp_native_speculative_accept(self, handle, n_accepted):
        self.calls.append(("accept", handle, int(n_accepted)))
        return True

    def llama_cpp_native_speculative_memory_seq_rm(self, handle, p0, p1):
        self.calls.append(("memory_seq_rm", handle, int(p0), int(p1)))
        return True

    def llama_cpp_native_speculative_memory_seq_add(
        self, handle, p0, p1, delta
    ):
        self.calls.append(
            ("memory_seq_add", handle, int(p0), int(p1), int(delta))
        )
        return True

    def llama_cpp_native_speculative_memory_clear(self, handle, clear_data):
        self.calls.append(("memory_clear", handle, bool(clear_data)))
        return True

    def llama_cpp_native_speculative_print_stats(self, handle):
        self.calls.append(("print_stats", handle))


@pytest.fixture
def fake_native(monkeypatch):
    native = _FakeNativeModule()
    load_calls = []

    def load_native_module():
        load_calls.append(True)
        return native

    monkeypatch.setattr(
        LlamaNativeSpeculativeDecoding,
        "_load_native_module",
        staticmethod(load_native_module),
    )
    return native, load_calls


@pytest.fixture(autouse=True)
def fake_draft_file_exists(monkeypatch):
    monkeypatch.setattr(
        "llama_cpp.llama_speculative.os.path.isfile", lambda path: bool(path)
    )


def test_native_speculative_configuration_is_lazy(fake_native):
    _native, load_calls = fake_native

    draft = LlamaNativeSpeculativeDecoding("draft.gguf")

    assert draft.is_native is True
    assert draft.spec_type == "draft-dflash"
    assert draft.max_draft_tokens == 15
    assert load_calls == []

    draft.close()
    assert load_calls == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"spec_type": "not-a-speculator"}, "spec_type"),
        ({"n_max": 0}, "n_max"),
        ({"n_min": -1}, "n_min"),
        ({"n_max": 3, "n_min": 4}, "n_min"),
        ({"p_min": -0.01}, "p_min"),
        ({"p_min": 1.01}, "p_min"),
        ({"n_gpu_layers": "some"}, "n_gpu_layers"),
    ],
)
def test_native_speculative_validates_configuration(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        LlamaNativeSpeculativeDecoding("draft.gguf", **kwargs)


def test_native_speculative_rejects_empty_model_path():
    with pytest.raises((TypeError, ValueError), match="model path"):
        LlamaNativeSpeculativeDecoding("")


@pytest.mark.parametrize(
    ("configured", "expected"),
    [("auto", -1), ("all", -2), (0, 0), (12, 12)],
)
def test_native_speculative_maps_gpu_layer_setting(
    fake_native, configured, expected
):
    native, _load_calls = fake_native
    draft = LlamaNativeSpeculativeDecoding(
        "draft.gguf", n_gpu_layers=configured
    )

    draft._bind(_FakeModel(), _FakeContext())

    assert native.init_params is not None
    assert native.init_params["model_path"] == b"draft.gguf"
    assert native.init_params["spec_type"] == b"draft-dflash"
    assert native.init_params["n_gpu_layers"] == expected
    assert native.init_params["n_max"] == 15
    draft.close()


def test_native_speculative_forwards_lifecycle_and_runtime_calls(fake_native):
    native, load_calls = fake_native
    draft = LlamaNativeSpeculativeDecoding(
        "draft.gguf",
        spec_type="draft-dflash",
        n_max=3,
        n_min=1,
        p_min=0.25,
    )

    draft._bind(_FakeModel(), _FakeContext())
    assert len(load_calls) == 1

    draft._begin([10, 20])
    draft._process_batch(_FakeBatch(n_tokens=3))

    result = draft(np.asarray([10, 20, 30], dtype=np.intc))
    assert result.dtype == np.dtype(np.intc)
    assert result.tolist() == [41, 42]

    draft.accept(1)
    assert draft.stats == {
        "draft_calls": 1,
        "accept_calls": 1,
        "drafted_tokens": 2,
        "accepted_tokens": 1,
        "acceptance_rate": 0.5,
        "mean_accepted_tokens": 1.0,
    }
    draft.memory_seq_rm(2, -1)
    draft.memory_seq_add(4, 8, -2)
    draft.clear()

    assert ("begin", native.handle, [10, 20]) in native.calls
    assert ("process", native.handle, 3) in native.calls
    assert (
        "draft",
        native.handle,
        2,
        30,
        [10, 20, 30],
        3,
    ) in native.calls
    assert ("accept", native.handle, 1) in native.calls
    assert ("memory_seq_rm", native.handle, 2, -1) in native.calls
    assert ("memory_seq_add", native.handle, 4, 8, -2) in native.calls
    assert ("memory_clear", native.handle, True) in native.calls

    draft.close()
    draft.close()
    assert native.calls.count(("free", native.handle)) == 1


def test_native_speculative_binding_is_idempotent_but_not_shareable(fake_native):
    native, load_calls = fake_native
    model = _FakeModel()
    context = _FakeContext()
    draft = LlamaNativeSpeculativeDecoding("draft.gguf")

    draft._bind(model, context)
    draft._bind(model, context)

    assert len(load_calls) == 1
    assert sum(call[0] == "init" for call in native.calls) == 1

    with pytest.raises(RuntimeError, match="cannot be shared"):
        draft._bind(_FakeModel(303), _FakeContext(404))

    draft.close()


def test_native_speculative_reports_initialization_error(fake_native):
    native, _load_calls = fake_native
    native.init_error = "incompatible target and draft"
    draft = LlamaNativeSpeculativeDecoding("draft.gguf")

    with pytest.raises(RuntimeError, match="incompatible target and draft"):
        draft._bind(_FakeModel(), _FakeContext())

    draft.close()
    assert not any(call[0] == "free" for call in native.calls)


def test_native_speculative_reports_draft_error(fake_native):
    native, _load_calls = fake_native
    native.draft_status = -1
    native.runtime_error = b"draft decode failed"
    draft = LlamaNativeSpeculativeDecoding("draft.gguf", n_max=3)
    draft._bind(_FakeModel(), _FakeContext())

    with pytest.raises(RuntimeError, match="draft decode failed"):
        draft(np.asarray([10, 20, 30], dtype=np.intc))

    draft.close()


@pytest.mark.parametrize(
    "operation",
    [
        lambda draft: draft._begin([1]),
        lambda draft: draft._process_batch(_FakeBatch(n_tokens=1)),
        lambda draft: draft(np.asarray([1], dtype=np.intc)),
        lambda draft: draft.accept(0),
        lambda draft: draft.memory_seq_rm(0, -1),
        lambda draft: draft.memory_seq_add(0, 1, -1),
        lambda draft: draft.clear(),
    ],
)
def test_native_speculative_requires_binding(operation):
    draft = LlamaNativeSpeculativeDecoding("draft.gguf")

    with pytest.raises(RuntimeError, match="bound|attach|initializ"):
        operation(draft)


class _EvalBatch:
    def __init__(self):
        self.batch = SimpleNamespace(n_tokens=0)
        self.logits = []

    def reset(self):
        self.batch.n_tokens = 0

    def add_sequence(self, token_array, pos_array, seq_ids, logits_array):
        self.batch.n_tokens = len(token_array)
        self.logits = list(logits_array)


class _EvalContext:
    def __init__(self, events, statuses):
        self.events = events
        self.statuses = iter(statuses)

    def clear_loras(self):
        pass

    def clear_cvec(self):
        pass

    def decode(self, batch):
        self.events.append(("decode", batch.batch.n_tokens))
        return next(self.statuses)


class _ProcessRecorder:
    def __init__(self, events):
        self.events = events

    def _process_batch(self, batch):
        self.events.append(("process", batch.n_tokens))


def _make_eval_only_llama(statuses):
    llm = object.__new__(Llama)
    events = []
    llm._ctx = _EvalContext(events, statuses)
    llm._batch = _EvalBatch()
    llm._native_speculative = _ProcessRecorder(events)
    llm._n_vocab = 256
    llm._n_ctx = 32
    llm.n_batch = 8
    llm.n_tokens = 0
    llm.input_ids = np.empty(32, dtype=np.intc)
    llm.scores = np.empty((1, 256), dtype=np.single)
    llm._logits_all = False
    llm.is_hybrid = False
    llm._hybrid_cache_mgr = None
    llm.verbose = False
    return llm, events


def test_eval_processes_native_features_only_after_successful_decode():
    llm, events = _make_eval_only_llama([1, 0, 0])

    llm.eval([1, 2, 3, 4], copy_logits=False, output_all=True)

    assert events == [
        ("decode", 4),
        ("decode", 2),
        ("process", 2),
        ("decode", 2),
        ("process", 2),
    ]
    assert llm._batch.logits == [True, True]
    assert llm.n_tokens == 4


def test_atomic_native_verification_never_downgrades_the_batch():
    llm, events = _make_eval_only_llama([1])

    with pytest.raises(RuntimeError, match="one successful target decode"):
        llm.eval(
            [1, 2, 3],
            copy_logits=False,
            output_all=True,
            require_atomic=True,
        )

    assert events == [("decode", 3)]


class _GenerationContext:
    def __init__(self):
        self.removals = []
        self.clear_count = 0
        self.logits = (ctypes.c_float * 256)()

    def memory_clear(self, data):
        self.clear_count += 1

    def memory_seq_rm(self, seq_id, p0, p1):
        self.removals.append((seq_id, p0, p1))
        return True

    def get_logits_ith(self, index):
        return ctypes.cast(self.logits, ctypes.POINTER(ctypes.c_float))


class _GenerationNative:
    max_draft_tokens = 2

    def __init__(self):
        self.begins = []
        self.accepted = []
        self.removals = []
        self.clear_count = 0

    def _begin(self, prompt):
        self.begins.append(np.asarray(prompt).tolist())

    def __call__(self, input_ids, **kwargs):
        return np.asarray([30, 40], dtype=np.intc)

    def accept(self, count):
        self.accepted.append(count)

    def memory_seq_rm(self, p0, p1):
        self.removals.append((p0, p1))

    def clear(self):
        self.clear_count += 1


class _GenerationSampler:
    outputs = []
    indices = []

    def __init__(self, params, model):
        self._outputs = iter(type(self).outputs)

    def sample(self, context, idx):
        type(self).indices.append(idx)
        return next(self._outputs)

    def accept(self, token, apply_grammar):
        pass

    def close(self):
        pass


def test_native_generation_samples_each_verification_row_and_rolls_back(
    monkeypatch,
):
    native = _GenerationNative()
    context = _GenerationContext()
    eval_calls = []

    llm = object.__new__(Llama)
    llm._native_speculative = native
    llm.draft_model = native
    llm._ctx = context
    llm._model = object()
    llm._hybrid_cache_mgr = None
    llm.is_hybrid = True  # Muse Glimmer is detected as hybrid because of SWA.
    llm.verbose = False
    llm.n_tokens = 0
    llm._n_ctx = 32
    llm._n_vocab = 256
    llm.input_ids = np.empty(32, dtype=np.intc)
    llm.scores = np.empty((1, 256), dtype=np.single)
    llm._logits_all = False
    llm._sampling_ctx = None
    llm._seed = 123
    llm._abort_event = threading.Event()

    def fake_eval(
        self,
        tokens,
        active_loras=None,
        control_vector=None,
        copy_logits=True,
        output_all=False,
        require_atomic=False,
    ):
        tokens = list(tokens)
        eval_calls.append((tokens, output_all, require_atomic))
        start = self.n_tokens
        self.input_ids[start : start + len(tokens)] = tokens
        self.n_tokens += len(tokens)

    llm.eval = MethodType(fake_eval, llm)

    _GenerationSampler.outputs = [20, 30, 99]
    _GenerationSampler.indices = []
    monkeypatch.setattr(llama_module, "LlamaSamplingContext", _GenerationSampler)

    generation = llm.generate([10], reset=True, temp=0.0)
    assert next(generation) == 20
    assert next(generation) == 30
    assert next(generation) == 99

    assert eval_calls == [
        ([10], False, False),
        ([20, 30, 40], True, True),
    ]
    assert all(isinstance(token, int) for token in eval_calls[1][0])
    assert _GenerationSampler.indices == [-1, -3, -2]
    assert native.begins == [[10]]
    assert native.accepted == [1]
    assert context.removals == [(0, 3, -1)]
    assert native.removals == [(3, -1)]
    assert llm.n_tokens == 3

    generation.close()


def test_native_generation_supports_token_stopping_criteria(monkeypatch):
    native = _GenerationNative()
    context = _GenerationContext()
    seen = []

    llm = object.__new__(Llama)
    llm._native_speculative = native
    llm.draft_model = native
    llm._ctx = context
    llm._model = object()
    llm._hybrid_cache_mgr = None
    llm.is_hybrid = True
    llm.verbose = False
    llm.n_tokens = 0
    llm._n_ctx = 32
    llm._n_vocab = 256
    llm.input_ids = np.empty(32, dtype=np.intc)
    llm.scores = np.empty((1, 256), dtype=np.single)
    llm._logits_all = False
    llm._sampling_ctx = None
    llm._seed = 123
    llm._abort_event = threading.Event()

    def fake_eval(
        self,
        tokens,
        active_loras=None,
        control_vector=None,
        copy_logits=True,
        output_all=False,
        require_atomic=False,
    ):
        tokens = list(tokens)
        start = self.n_tokens
        self.input_ids[start : start + len(tokens)] = tokens
        self.n_tokens += len(tokens)

    def stop_on_30(input_ids, logits):
        seen.append((input_ids.tolist(), logits.shape))
        return int(input_ids[-1]) == 30

    llm.eval = MethodType(fake_eval, llm)
    _GenerationSampler.outputs = [20, 30]
    _GenerationSampler.indices = []
    monkeypatch.setattr(llama_module, "LlamaSamplingContext", _GenerationSampler)

    generation = llm.generate(
        [10],
        reset=True,
        temp=0.0,
        stopping_criteria=llama_module.StoppingCriteriaList([stop_on_30]),
    )

    assert next(generation) == 20
    with pytest.raises(StopIteration):
        next(generation)

    assert seen == [([10, 20], (256,)), ([10, 20, 30], (256,))]
    assert native.accepted == [1]
    assert context.removals == [(0, 3, -1)]
    assert native.removals == [(3, -1)]
