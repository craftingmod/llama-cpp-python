import abc
import collections
import ctypes
import os

from typing import Any, DefaultDict, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt


class LlamaDraftModel(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self, input_ids: npt.NDArray[np.intc], /, **kwargs: Any
    ) -> npt.NDArray[np.intc]:
        raise NotImplementedError()


class LlamaNativeSpeculativeDecoding(LlamaDraftModel):
    """Experimental llama.cpp/common GGUF draft-model integration.

    The draft model is loaded lazily when this object is attached to ``Llama``.
    This first implementation intentionally supports one text sequence and the
    DFlash/DSpark block-diffusion implementations only.
    """

    is_native = True
    _SUPPORTED_TYPES = {"draft-dflash", "draft-dspark"}

    def __init__(
        self,
        model_path: str,
        *,
        spec_type: Literal["draft-dflash", "draft-dspark"] = "draft-dflash",
        n_max: int = 15,
        n_min: int = 0,
        p_min: float = 0.0,
        n_gpu_layers: Union[int, Literal["auto", "all"]] = "auto",
        n_threads: Optional[int] = None,
        n_threads_batch: Optional[int] = None,
        type_k: Optional[int] = None,
        type_v: Optional[int] = None,
        flash_attn_type: int = -1,
        offload_kqv: bool = True,
        op_offload: bool = True,
        kv_unified: bool = False,
        no_perf: bool = False,
        verbose: bool = True,
    ) -> None:
        if not os.path.isfile(model_path):
            raise ValueError(f"Draft model path does not exist: {model_path}")

        normalized_type = spec_type.strip().lower()
        if normalized_type not in self._SUPPORTED_TYPES:
            raise ValueError(
                "spec_type must be 'draft-dflash' or 'draft-dspark' in this "
                "experimental implementation"
            )
        if n_max <= 0:
            raise ValueError("n_max must be greater than zero")
        if n_min < 0 or n_min > n_max:
            raise ValueError("n_min must be between zero and n_max")
        if not 0.0 <= p_min <= 1.0:
            raise ValueError("p_min must be between zero and one")
        if n_threads is not None and n_threads <= 0:
            raise ValueError("n_threads must be greater than zero")
        if n_threads_batch is not None and n_threads_batch <= 0:
            raise ValueError("n_threads_batch must be greater than zero")

        self.model_path = os.fspath(model_path)
        self.spec_type = normalized_type
        self.n_max = int(n_max)
        self.n_min = int(n_min)
        self.p_min = float(p_min)
        self.n_gpu_layers = self._parse_n_gpu_layers(n_gpu_layers)
        self.n_threads = n_threads
        self.n_threads_batch = n_threads_batch
        # GGML_TYPE_F16. Keeping the numeric default here avoids importing the
        # heavyweight ctypes backend before Llama attaches this object.
        self.type_k = 1 if type_k is None else int(type_k)
        self.type_v = 1 if type_v is None else int(type_v)
        self.flash_attn_type = int(flash_attn_type)
        self.offload_kqv = bool(offload_kqv)
        self.op_offload = bool(op_offload)
        self.kv_unified = bool(kv_unified)
        self.no_perf = bool(no_perf)
        self.verbose = bool(verbose)

        self._native: Any = None
        self._handle: Any = None
        self._bound_model: Any = None
        self._bound_context: Any = None
        self._closed = False
        self._last_draft_len = 0
        self._draft_calls = 0
        self._accept_calls = 0
        self._drafted_tokens = 0
        self._accepted_tokens = 0

    @property
    def max_draft_tokens(self) -> int:
        return self.n_max

    @property
    def stats(self) -> Dict[str, Union[int, float]]:
        """Return lifetime draft and acceptance counters for this instance."""
        acceptance_rate = (
            self._accepted_tokens / self._drafted_tokens
            if self._drafted_tokens > 0
            else 0.0
        )
        mean_accepted = (
            self._accepted_tokens / self._accept_calls
            if self._accept_calls > 0
            else 0.0
        )
        return {
            "draft_calls": self._draft_calls,
            "accept_calls": self._accept_calls,
            "drafted_tokens": self._drafted_tokens,
            "accepted_tokens": self._accepted_tokens,
            "acceptance_rate": acceptance_rate,
            "mean_accepted_tokens": mean_accepted,
        }

    @staticmethod
    def _parse_n_gpu_layers(value: Union[int, str]) -> int:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized == "auto":
                return -1
            if normalized == "all":
                return -2
            try:
                return int(normalized)
            except ValueError as exc:
                raise ValueError(
                    "n_gpu_layers must be an int, 'auto', or 'all'"
                ) from exc
        if isinstance(value, int):
            return value
        raise TypeError("n_gpu_layers must be an int, 'auto', or 'all'")

    @staticmethod
    def _load_native_module() -> Any:
        try:
            from . import llama_speculative_cpp
        except (AttributeError, ImportError, OSError, RuntimeError) as exc:
            raise RuntimeError(
                "Native speculative decoding is unavailable. Rebuild/reinstall "
                "llama-cpp-python from this experimental branch so llama-common "
                "contains the native speculative bridge."
            ) from exc
        return llama_speculative_cpp

    def _require_handle(self) -> Tuple[Any, Any]:
        if self._closed:
            raise RuntimeError("Native speculative decoder has been closed")
        if self._native is None or self._handle is None:
            raise RuntimeError(
                "Native speculative decoder is not attached to a Llama instance"
            )
        return self._native, self._handle

    def _last_error(self) -> str:
        if self._native is None or self._handle is None:
            return "native speculative decoder is not initialized"
        message = self._native.llama_cpp_native_speculative_last_error(self._handle)
        return message.decode("utf-8", errors="replace") if message else "unknown error"

    def _bind(self, model: Any, context: Any) -> None:
        if self._closed:
            raise RuntimeError("Cannot attach a closed native speculative decoder")
        if self._handle is not None:
            if model is self._bound_model and context is self._bound_context:
                return
            raise RuntimeError(
                "A native speculative decoder instance cannot be shared by multiple Llama instances"
            )

        native = self._load_native_module()
        params = native.llama_cpp_native_speculative_params()
        params.struct_size = ctypes.sizeof(native.llama_cpp_native_speculative_params)
        params.model_path = os.fsencode(self.model_path)
        params.spec_type = self.spec_type.encode("ascii")
        params.n_gpu_layers = self.n_gpu_layers
        params.n_ctx = int(context.n_ctx())
        params.n_batch = int(context.n_batch())
        params.n_ubatch = int(context.n_ubatch())
        params.n_threads = int(self.n_threads or context.params.n_threads)
        params.n_threads_batch = int(
            self.n_threads_batch or context.params.n_threads_batch
        )
        params.n_max = self.n_max
        params.n_min = self.n_min
        params.p_min = self.p_min
        params.cache_type_k = self.type_k
        params.cache_type_v = self.type_v
        params.flash_attn_type = self.flash_attn_type
        params.offload_kqv = self.offload_kqv
        params.op_offload = self.op_offload
        params.kv_unified = self.kv_unified
        params.no_perf = self.no_perf

        error = ctypes.create_string_buffer(1024)
        handle = native.llama_cpp_native_speculative_init(
            model.model,
            context.ctx,
            ctypes.byref(params),
            error,
            len(error),
        )
        if not handle:
            message = error.value.decode("utf-8", errors="replace")
            raise RuntimeError(
                "Failed to initialize native speculative decoder: "
                + (message or "unknown error")
            )

        self._native = native
        self._handle = handle
        self._bound_model = model
        self._bound_context = context

    @staticmethod
    def _token_buffer(tokens: npt.ArrayLike) -> Tuple[npt.NDArray[np.int32], Any]:
        array = np.ascontiguousarray(tokens, dtype=np.int32)
        pointer = array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        return array, pointer

    def _begin(self, prompt_tokens: npt.ArrayLike) -> None:
        native, handle = self._require_handle()
        prompt, pointer = self._token_buffer(prompt_tokens)
        if not native.llama_cpp_native_speculative_begin(
            handle, pointer, prompt.size
        ):
            raise RuntimeError(
                f"Native speculative begin failed: {self._last_error()}"
            )

    def _process_batch(self, batch: Any) -> None:
        native, handle = self._require_handle()
        if not native.llama_cpp_native_speculative_process(
            handle, ctypes.byref(batch)
        ):
            raise RuntimeError(
                f"Native speculative batch processing failed: {self._last_error()}"
            )

    def __call__(
        self, input_ids: npt.NDArray[np.intc], /, **kwargs: Any
    ) -> npt.NDArray[np.intc]:
        native, handle = self._require_handle()
        prompt, pointer = self._token_buffer(input_ids)
        if prompt.size == 0:
            raise ValueError("Native speculative decoding requires at least one input token")

        capacity = min(
            self.n_max,
            int(kwargs.get("max_tokens", self.n_max)),
        )
        if capacity <= 0:
            return np.empty(0, dtype=np.intc)

        output = np.empty(capacity, dtype=np.int32)
        count = native.llama_cpp_native_speculative_draft(
            handle,
            prompt.size - 1,
            int(prompt[-1]),
            pointer,
            prompt.size,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            capacity,
        )
        if count < 0:
            raise RuntimeError(f"Native speculative draft failed: {self._last_error()}")
        self._last_draft_len = int(count)
        self._draft_calls += 1
        self._drafted_tokens += self._last_draft_len
        return output[:count].astype(np.intc, copy=False)

    def accept(self, n_accepted: int) -> None:
        native, handle = self._require_handle()
        if n_accepted < 0 or n_accepted > self._last_draft_len:
            raise ValueError("n_accepted must be within the last native draft")
        if not native.llama_cpp_native_speculative_accept(handle, n_accepted):
            raise RuntimeError(f"Native speculative accept failed: {self._last_error()}")
        self._accept_calls += 1
        self._accepted_tokens += n_accepted
        self._last_draft_len = 0

    def memory_seq_rm(self, p0: int, p1: int) -> None:
        native, handle = self._require_handle()
        if not native.llama_cpp_native_speculative_memory_seq_rm(handle, p0, p1):
            raise RuntimeError(
                f"Native draft-context rollback failed: {self._last_error()}"
            )

    def memory_seq_add(self, p0: int, p1: int, delta: int) -> None:
        native, handle = self._require_handle()
        if not native.llama_cpp_native_speculative_memory_seq_add(
            handle, p0, p1, delta
        ):
            raise RuntimeError(
                f"Native draft-context shift failed: {self._last_error()}"
            )

    def clear(self) -> None:
        native, handle = self._require_handle()
        if not native.llama_cpp_native_speculative_memory_clear(handle, True):
            raise RuntimeError(
                f"Native draft-context clear failed: {self._last_error()}"
            )
        self._last_draft_len = 0

    def print_stats(self) -> None:
        native, handle = self._require_handle()
        stats = self.stats
        print(
            "native speculative stats: "
            f"draft calls={stats['draft_calls']}, "
            f"accept calls={stats['accept_calls']}, "
            f"drafted tokens={stats['drafted_tokens']}, "
            f"accepted tokens={stats['accepted_tokens']}, "
            f"token acceptance={stats['acceptance_rate']:.1%}, "
            f"mean accepted/call={stats['mean_accepted_tokens']:.2f}",
            flush=True,
        )
        native.llama_cpp_native_speculative_print_stats(handle)

    def close(self) -> None:
        if self._closed:
            return
        if self._native is not None and self._handle is not None:
            self._native.llama_cpp_native_speculative_free(self._handle)
        self._handle = None
        self._bound_context = None
        self._bound_model = None
        self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class LlamaNGramMapDecoding(LlamaDraftModel):
    """
    Fast model-free speculative decoder based on prompt n-gram lookup.

    It supports two modes:

    - "k":
        Key-only mode. Stores n-gram key -> history positions.
        This is memory-efficient and similar to llama.cpp's ngram-map-k behavior.

    - "k4v":
        Key-to-value mode. Stores n-gram key -> continuation tokens.
        This uses more memory, but can return cached continuations directly.

    This class does not use a draft model. It only speculates from already verified
    token history. Therefore, rejected tokens are handled naturally when the next
    `input_ids` is passed in.

    Aligned with llama.cpp's underlying ngram-map k/k4v algorithm.
    """

    def __init__(
        self,
        ngram_size: int = 3,
        num_pred_tokens: int = 10,
        mode: Literal["k", "k4v"] = "k",
        min_hits: int = 2,
        max_entries_per_key: Optional[int] = None,
        sync_check_tokens: int = 16,
    ) -> None:
        """
        Args:
            ngram_size:
                Number of tokens used as the lookup key.

            num_pred_tokens:
                Maximum number of draft tokens to return.

            mode:
                "k" stores only matched positions.
                "k4v" stores matched continuation values directly.

            min_hits:
                Minimum number of historical matches required before returning a draft.
                Use 1 for maximum recall. Use >1 to reduce low-confidence drafts.

            max_entries_per_key:
                Optional memory cap per n-gram key.
                When set, only the most recent entries are kept.
                For k4v mode, setting max_entries_per_key is strongly recommended.

            sync_check_tokens:
                Number of trailing tokens used to verify whether the new input is an
                incremental append of the previous input. This avoids expensive full
                prefix comparison while still detecting most rollback/prompt-switch cases.
        """
        if ngram_size <= 0:
            raise ValueError("ngram_size must be greater than 0")
        if num_pred_tokens <= 0:
            raise ValueError("num_pred_tokens must be greater than 0")
        if min_hits <= 0:
            raise ValueError("min_hits must be greater than 0")
        if max_entries_per_key is not None and max_entries_per_key <= 0:
            raise ValueError("max_entries_per_key must be None or greater than 0")
        if sync_check_tokens <= 0:
            raise ValueError("sync_check_tokens must be greater than 0")

        mode = mode.lower()
        if mode not in ("k", "k4v"):
            raise ValueError("mode must be either 'k' or 'k4v'")

        self.ngram_size = int(ngram_size)
        self.num_pred_tokens = int(num_pred_tokens)
        self.mode = mode
        self.min_hits = int(min_hits)
        self.sync_check_tokens = int(sync_check_tokens)

        if mode == "k4v" and max_entries_per_key is None:
            max_entries_per_key = 8
        self.max_entries_per_key = max_entries_per_key

        self._history: List[int] = []

        # In "k" mode:
        #   key -> [position, position, ...]
        self._map_k: DefaultDict[Tuple[int, ...], List[int]] = collections.defaultdict(list)

        # In "k4v" mode:
        #   key -> {position: continuation}
        #
        # A dict is used so that recent entries can be refreshed when more continuation
        # tokens become available. Draft selection is based on continuation frequency,
        # not just the most recent continuation.
        self._map_k4v: DefaultDict[
            Tuple[int, ...], Dict[int, Tuple[int, ...]]
        ] = collections.defaultdict(dict)

        # Acceptance feedback, aligned with llama.cpp's ngram-map behavior:
        # accept(n) stores how many tokens were accepted for the key/value used by
        # the previous draft and limits the future draft length for that key/value.
        self._accepted_k: Dict[Tuple[int, ...], int] = {}
        self._accepted_k4v: DefaultDict[
            Tuple[int, ...], Dict[Tuple[int, ...], int]
        ] = collections.defaultdict(dict)

        self._last_draft_key: Optional[Tuple[int, ...]] = None
        self._last_draft_value: Optional[Tuple[int, ...]] = None

        self._closed = False
        self._last_draft_len = 0

    def clear(self) -> None:
        """
        Clear token history and indexes.

        Use this when starting a completely unrelated generation while keeping the
        decoder instance reusable.
        """
        self._history.clear()
        self._map_k.clear()
        self._map_k4v.clear()
        self._accepted_k.clear()
        self._accepted_k4v.clear()
        self._last_draft_key = None
        self._last_draft_value = None
        self._last_draft_len = 0

    def close(self) -> None:
        """
        Release internal memory.

        This class does not own native memory, but clearing large Python containers
        explicitly is still useful for long-running applications.
        """
        self.clear()
        self._closed = True

    def __del__(self) -> None:
        # Best-effort cleanup. Program correctness must not depend on __del__.
        try:
            self.close()
        except Exception:
            pass

    def accept(self, n_accepted: int) -> None:
        """
        Notify how many draft tokens were accepted by the target model.

        The accepted length is written back to the key/value used by the previous
        draft. Future drafts for the same key/value are truncated to this accepted
        length, matching llama.cpp's ngram-map feedback loop.
        """
        if n_accepted < 0:
            raise ValueError("n_accepted must be non-negative")

        if self._last_draft_key is None or self._last_draft_len <= 0:
            return

        accepted = min(int(n_accepted), self._last_draft_len)

        if self.mode == "k":
            self._accepted_k[self._last_draft_key] = accepted
        else:
            if self._last_draft_value is not None:
                self._accepted_k4v[self._last_draft_key][self._last_draft_value] = accepted

        self._last_draft_key = None
        self._last_draft_value = None
        self._last_draft_len = 0

    def _sync_and_index(self, input_ids: npt.NDArray[np.intc]) -> None:
        """
        Synchronize internal history with input_ids and update the n-gram index.

        The index intentionally stores only n-grams that have at least one continuation
        token. This prevents the current tail n-gram from matching itself and returning
        an empty draft.
        """
        if self._closed:
            raise RuntimeError("LlamaNGramMapDecoding is closed")

        tokens = np.asarray(input_ids, dtype=np.intc).reshape(-1).tolist()

        old_len = len(self._history)
        new_len = len(tokens)

        if new_len == 0:
            self.clear()
            return

        # Fast path: identical input, no update needed.
        if new_len == old_len:
            if self._history == tokens:
                return

        # Incremental append path.
        is_append = False
        if old_len > 0 and new_len > old_len:
            check_len = min(old_len, max(self.ngram_size, self.sync_check_tokens))
            is_append = self._history[old_len - check_len : old_len] == tokens[
                old_len - check_len : old_len
            ]

        if is_append:
            # Append only new tokens.
            self._history.extend(tokens[old_len:])

            if self.mode == "k":
                # Only newly-valid keys need to be added.
                start = max(0, old_len - self.ngram_size)
            else:
                # K4V must also refresh recent keys because their continuation values
                # can grow as new tokens are appended.
                start = max(0, old_len - self.ngram_size - self.num_pred_tokens + 1)
        else:
            # Rollback, prompt switch, truncation, or unsafe mutation.
            self.clear()
            self._history.extend(tokens)
            start = 0

        # Only index keys that have at least one token after the key.
        # Valid pos satisfies:
        #   pos + ngram_size < len(history)
        end = max(0, len(self._history) - self.ngram_size)

        if start >= end:
            return

        if self.mode == "k":
            for pos in range(start, end):
                key = tuple(self._history[pos : pos + self.ngram_size])
                bucket = self._map_k[key]

                if not bucket or bucket[-1] != pos:
                    bucket.append(pos)

                if (
                    self.max_entries_per_key is not None
                    and len(bucket) > self.max_entries_per_key
                ):
                    del bucket[: len(bucket) - self.max_entries_per_key]

        else:
            for pos in range(start, end):
                key_start = pos
                value_start = pos + self.ngram_size
                value_end = value_start + self.num_pred_tokens

                # K4V tracks fixed-size continuation m-grams. Partial tail values are
                # intentionally skipped so frequency statistics remain comparable.
                if value_end > len(self._history):
                    continue

                key = tuple(self._history[key_start:value_start])
                value = tuple(self._history[value_start:value_end])

                bucket = self._map_k4v[key]
                bucket[pos] = value

                if (
                    self.max_entries_per_key is not None
                    and len(bucket) > self.max_entries_per_key
                ):
                    # Keep the most recent positions.
                    for old_pos in sorted(bucket)[: len(bucket) - self.max_entries_per_key]:
                        del bucket[old_pos]

    def __call__(
        self, input_ids: npt.NDArray[np.intc], /, **kwargs: Any
    ) -> npt.NDArray[np.intc]:
        """
        Generate draft tokens from verified token history.

        Args:
            input_ids:
                Complete verified token sequence so far.

        Returns:
            np.ndarray[np.intc]:
                Predicted draft tokens. Empty array means no reliable match was found.
        """
        _ = kwargs

        self._sync_and_index(input_ids)
        self._last_draft_key = None
        self._last_draft_value = None
        self._last_draft_len = 0

        if len(self._history) < self.ngram_size:
            return np.array([], dtype=np.intc)

        search_key = tuple(self._history[-self.ngram_size :])

        if self.mode == "k":
            positions = self._map_k.get(search_key)
            if not positions:
                return np.array([], dtype=np.intc)

            # Key-only mode follows llama.cpp's ngram-map-k behavior: once a key
            # match is found, draft from the latest valid match. min_hits is not
            # used as a confidence gate for key-only mode.
            draft: List[int] = []
            accepted_limit = self._accepted_k.get(search_key, self.num_pred_tokens)
            if accepted_limit <= 0:
                return np.array([], dtype=np.intc)

            for pos in reversed(positions):
                start = pos + self.ngram_size
                if start < len(self._history):
                    end = min(start + accepted_limit, len(self._history))
                    draft = self._history[start:end]
                    break

            self._last_draft_key = search_key

        else:
            values = self._map_k4v.get(search_key)
            if not values or len(values) < self.min_hits:
                return np.array([], dtype=np.intc)

            # K4V mode chooses the most frequent continuation m-gram rather than the
            # latest one. If the strongest continuation is not at least twice as
            # frequent as all other continuations combined, skip drafting.
            counts = collections.Counter(values.values())
            best_value, best_count = counts.most_common(1)[0]
            other_count = sum(counts.values()) - best_count

            if other_count > 0 and best_count < 2 * other_count:
                return np.array([], dtype=np.intc)

            accepted_limit = self._accepted_k4v[search_key].get(
                best_value, self.num_pred_tokens
            )
            if accepted_limit <= 0:
                return np.array([], dtype=np.intc)

            draft = list(best_value[:accepted_limit])
            self._last_draft_key = search_key
            self._last_draft_value = best_value

        self._last_draft_len = len(draft)
        if self._last_draft_len <= 0:
            self._last_draft_key = None
            self._last_draft_value = None
            return np.array([], dtype=np.intc)

        return np.asarray(draft, dtype=np.intc)


# Legacy Numpy sliding window implementation
# Fast in some cases, but may degrade output quality.
# Not recommended for production.
class LlamaPromptLookupDecoding(LlamaDraftModel):
    """
    Stateless speculative decoding based on Numpy sliding window
    Warning: High computational overhead for long contexts.

    Based on https://github.com/apoorvumang/prompt-lookup-decoding
    """

    def __init__(self, max_ngram_size: int = 3, num_pred_tokens: int = 10):
        """
        Initializes the legacy sliding window speculative decoder.

        Args:
            max_ngram_size (int): The maximum n-gram size to search for. Defaults to 3.
            num_pred_tokens (int): The maximum number of tokens to predict. Defaults to 10.
        """
        self.max_ngram_size = max_ngram_size
        self.num_pred_tokens = num_pred_tokens

    @staticmethod
    def find_candidate_pred_tokens(
        input_ids: npt.NDArray[np.intc],
        max_ngram_size: int,
        num_pred_tokens: int,
    ):
        """
        Linearly scans the input_ids using sliding windows to find pattern matches.

        Args:
            input_ids (npt.NDArray[np.intc]): The complete sequence of token IDs.
            max_ngram_size (int): Maximum size of the n-gram window.
            num_pred_tokens (int): Maximum draft tokens to return.

        Returns:
            npt.NDArray[np.intc]: The predicted draft tokens.
        """
        input_length = input_ids.shape[0]

        for ngram_size in range(min(max_ngram_size, input_length - 1), 0, -1):
            # Create sliding windows of size ngram_size
            windows = np.lib.stride_tricks.sliding_window_view(input_ids, (ngram_size,))

            # Convert ngram to an array for comparison
            ngram_array = input_ids[-ngram_size:]

            # Find where the windows match the ngram
            matches = np.all(windows == ngram_array, axis=1)

            # Get the indices of matches
            match_indices = np.nonzero(matches)[0]

            # Iterate through match indices to find a valid continuation
            for idx in match_indices:
                start_idx = idx + ngram_size
                end_idx = start_idx + num_pred_tokens
                end_idx = min(end_idx, input_length)

                if start_idx < end_idx:
                    return input_ids[start_idx:end_idx]

        # If no match is found, return an empty array
        return np.array([], dtype=np.intc)

    def __call__(
        self, input_ids: npt.NDArray[np.intc], /, **kwargs: Any
    ) -> npt.NDArray[np.intc]:
        """Generates draft tokens using the legacy sliding window search."""
        return self.find_candidate_pred_tokens(
            input_ids=input_ids,
            max_ngram_size=self.max_ngram_size,
            num_pred_tokens=self.num_pred_tokens,
        )
