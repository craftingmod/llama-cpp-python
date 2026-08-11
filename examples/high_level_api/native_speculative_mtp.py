"""Text-only Gemma 4 MTP smoke test using an external assistant GGUF."""

import argparse

from llama_cpp import Llama, llama_flash_attn_type
from llama_cpp.llama_speculative import LlamaNativeSpeculativeDecoding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="Gemma 4 target GGUF")
    parser.add_argument("assistant", help="Matching gemma4-assistant MTP GGUF")
    parser.add_argument(
        "--prompt",
        default="Explain why speculative decoding can improve inference speed.",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--n-max", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-batch", type=int, default=512)
    parser.add_argument("--n-gpu-layers", default="all")
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.requests <= 0:
        parser.error("--requests must be greater than zero")

    draft = LlamaNativeSpeculativeDecoding(
        model_path=args.assistant,
        spec_type="draft-mtp",
        n_gpu_layers=args.n_gpu_layers,
        n_max=args.n_max,
        verbose=args.verbose,
    )
    llm = None
    try:
        llm = Llama(
            model_path=args.target,
            draft_model=draft,
            n_gpu_layers=args.n_gpu_layers,
            n_ctx=args.n_ctx,
            n_batch=args.n_batch,
            n_ubatch=min(args.n_batch, 512),
            flash_attn_type=(
                llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_ENABLED
            ),
            verbose=args.verbose,
        )

        for request_index in range(args.requests):
            response = llm.create_chat_completion(
                messages=[{"role": "user", "content": args.prompt}],
                max_tokens=args.max_tokens,
                temperature=0.0,
                seed=1234,
            )
            content = response["choices"][0]["message"]["content"]
            print(f"\n--- request {request_index + 1} ---\n{content}")

        print(f"\nPython counters: {draft.stats}")
        draft.print_stats()
    finally:
        if llm is not None:
            llm.close()
        draft.close()


if __name__ == "__main__":
    main()
