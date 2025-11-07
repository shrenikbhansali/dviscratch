"""Generate answers with local models."""
import os
import argparse
from fastchat.utils import str_to_torch_dtype
from kangaroo.cli_utils import str2bool

from evaluation.eval import run_eval, reorg_answer_file
from evaluation.sharegpt_eval import ShareGPTConfig, stream_sharegpt_answers

from transformers import AutoModelForCausalLM, AutoTokenizer

def baseline_forward(inputs, model, tokenizer, max_new_tokens, temperature=0.0, do_sample=False):
    input_ids = inputs.input_ids
    output_ids = model.generate(
        input_ids,
        do_sample=do_sample,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    new_token = len(output_ids[0][len(input_ids[0]):])
    idx = new_token - 1
    accept_length_list = [1] * new_token
    return output_ids, new_token, idx, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end",
        type=int,
        help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for medusa sampling.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )
    parser.add_argument(
        "--sharegpt-jsonl",
        type=str,
        default=None,
        help="Optional ShareGPT JSONL to stream prompts from instead of MT-Bench.",
    )
    parser.add_argument("--sharegpt-max-samples", type=int, default=0, help="Limit ShareGPT samples processed.")
    parser.add_argument("--sharegpt-skip-samples", type=int, default=0, help="Skip this many ShareGPT samples first.")
    parser.add_argument(
        "--sharegpt-keep-system",
        type=str2bool,
        default=False,
        help="Keep ShareGPT system prompts in the constructed prompt.",
    )
    parser.add_argument(
        "--sharegpt-use-last-turn",
        type=str2bool,
        default=True,
        help="Emit only the last assistant reply per conversation.",
    )
    parser.add_argument(
        "--sharegpt-max-src-len",
        type=int,
        default=2048,
        help="Truncate ShareGPT prompts to this many characters.",
    )
    parser.add_argument(
        "--sharegpt-max-tgt-len",
        type=int,
        default=512,
        help="Truncate ShareGPT reference answers to this many characters.",
    )
    parser.add_argument(
        "--sharegpt-output",
        type=str,
        default=None,
        help="Path to save ShareGPT streaming results (defaults under data/sharegpt_runs).",
    )

    args = parser.parse_args()

    question_file = f"data/question.jsonl"

    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.temperature > 0:
        do_sample = True
    else:
        do_sample = False

    if args.sharegpt_jsonl:
        safe_model = args.model_id.replace("/", "_")
        default_out = os.path.join("data", "sharegpt_runs", f"{safe_model}_vanilla.jsonl")
        sharegpt_output = args.sharegpt_output or default_out
        cfg = ShareGPTConfig(
            path=args.sharegpt_jsonl,
            max_samples=args.sharegpt_max_samples,
            skip_samples=args.sharegpt_skip_samples,
            max_src_len=args.sharegpt_max_src_len,
            max_tgt_len=args.sharegpt_max_tgt_len,
            keep_system=args.sharegpt_keep_system,
            use_last_turn=args.sharegpt_use_last_turn,
        )
        stats = stream_sharegpt_answers(
            model=model,
            tokenizer=tokenizer,
            forward_func=baseline_forward,
            output_path=sharegpt_output,
            max_new_tokens=args.max_new_tokens,
            sharegpt_cfg=cfg,
            forward_kwargs={
                "temperature": args.temperature,
                "do_sample": do_sample,
            },
        )
        print(
            f"[ShareGPT][baseline] completed {stats['processed']} samples "
            f"(tokens/sec={stats['tokens_per_second']:.2f}, avg_wall={stats['avg_wall_time']:.2f}s)"
        )
    else:
        assert not args.answer_file
        os.makedirs(f"data/{args.bench_name}/{args.model_id}", exist_ok=True)

        for run in range(3):
            answer_file = f"data/{args.bench_name}/{args.model_id}/{run}.jsonl"
            print(f"Output to {answer_file}")
            run_eval(
                model=model,
                tokenizer=tokenizer,
                forward_func=baseline_forward,
                model_id=args.model_id,
                question_file=question_file,
                question_begin=args.question_begin,
                question_end=args.question_end,
                answer_file=answer_file,
                max_new_tokens=args.max_new_tokens,
                num_choices=args.num_choices,
                num_gpus_per_model=args.num_gpus_per_model,
                num_gpus_total=args.num_gpus_total,
                temperature=args.temperature,
                do_sample=do_sample,
            )
            reorg_answer_file(answer_file)
