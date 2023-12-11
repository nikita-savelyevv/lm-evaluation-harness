import shutil
import traceback

from dataclasses import dataclass
import argparse
import json
import logging
from datetime import datetime
from time import time, sleep
import random

from lm_eval import evaluator
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.intel import OVModelForCausalLM

logging.getLogger("openai").setLevel(logging.WARNING)

import openvino.runtime as ov
from openvino import Core
import openvino
from pathlib import Path
core = Core()


LOGS_DIR = Path("./logs_compress")

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", required=True)
    # parser.add_argument(
    #     "--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS)
    # )
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=100)
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="Maximal batch size to try with --batch_size auto",
    )
    parser.add_argument("--device", type=str, default='auto')
    parser.add_argument("--output_path", default=None)
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true", default=True)
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)
    parser.add_argument("--delete_ir_cache", action="store_true", default=False)
    parser.add_argument("--do_eval", action="store_true", default=False)

    return parser.parse_args()

@dataclass
class ExpDesc:
    model_path: str
    device: str = "auto"


def main():
    args = parse_args()

    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    # if args.tasks is None:
    #     task_names = tasks.ALL_TASKS
    # else:
    #     task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    # print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    use_pkv = True
    descs = [
        # ExpDesc(model_path="fp16_calibration_models/red-pajama-3b-chat/FP16", device="cpu"),
        # ExpDesc(model_path="fp16_calibration_models/red-pajama-3b-chat/FP16", device="gpu"),
        # ExpDesc(model_path="fp16_calibration_models/red-pajama-3b-chat/FP16_calibrated", device="cpu"),
        # ExpDesc(model_path="fp16_calibration_models/red-pajama-3b-chat/FP16_calibrated", device="gpu"),
        # ExpDesc(model_path="fp16_calibration_models/red-pajama-3b-chat/FP16_calibrated_0.05", device="gpu"),
        # ExpDesc(model_path="fp16_calibration_models/red-pajama-3b-chat/FP16_calibrated_0.10", device="gpu"),
        # ExpDesc(model_path="fp16_calibration_models/red-pajama-3b-chat/FP16_calibrated_0.20", device="gpu"),
        # ExpDesc(model_path="fp16_calibration_models/red-pajama-3b-chat/FP16_calibrated_0.30", device="gpu"),
        #
        ExpDesc(model_path="fp16_calibration_models/red-pajama-3b-chat/INT8_compressed_weights", device="cpu"),
        # ExpDesc(model_path="fp16_calibration_models/red-pajama-3b-chat/INT8_compressed_weights", device="gpu"),
        # ExpDesc(model_path="fp16_calibration_models/red-pajama-3b-chat/INT8_compressed_weights_calibrated", device="cpu"),
        # ExpDesc(model_path="fp16_calibration_models/red-pajama-3b-chat/INT8_compressed_weights_calibrated", device="gpu"),
        # ExpDesc(model_path="fp16_calibration_models/red-pajama-3b-chat/INT8_compressed_weights_calibrated_0.05", device="gpu"),
        # ExpDesc(model_path="fp16_calibration_models/red-pajama-3b-chat/INT8_compressed_weights_calibrated_0.10", device="gpu"),
        # ExpDesc(model_path="fp16_calibration_models/red-pajama-3b-chat/INT8_compressed_weights_calibrated_0.20", device="gpu"),
        # ExpDesc(model_path="fp16_calibration_models/red-pajama-3b-chat/INT8_compressed_weights_calibrated_0.30", device="gpu"),

        # ExpDesc(model_path="fp16_calibration_models/red-pajama-3b-chat/FP32", device="cpu"),
        # ExpDesc(model_path="fp16_calibration_models/red-pajama-3b-chat/FP32", device="gpu"),
    ]

    all_results_paths = []
    for desc in descs:
        try:
            model_path = desc.model_path
            printable_desc = json.dumps(desc.__dict__,  indent=4)
            print(f"Started experiment {printable_desc}\n")
            random.seed(42)

            log_dir = Path(model_path) / (f"evaluation_{desc.device}_" + datetime.now().strftime("%b%d_%H-%M-%S"))
            log_dir.mkdir(parents=True, exist_ok=True)
            with (log_dir / 'args.json').open('w') as f:
                f.write(printable_desc)

            # if not ir_path.exists():
            #     ov_model = OVModelForCausalLM.from_pretrained(model_id, use_cache=use_pkv, trust_remote_code=True,
            #                                                   from_transformers=True)
            #     ov_model.save_pretrained(ir_cache_dir)
            #     del ov_model
            #     gc.collect()

            start_time = time()
            device = desc.device if args.device == "auto" else args.device
            results = evaluator.simple_evaluate(
                model='optimum-causal',
                model_args=f'pretrained={model_path}',
                tasks=['lambada_openai'],
                num_fewshot=args.num_fewshot,
                batch_size=args.batch_size,
                max_batch_size=args.max_batch_size,
                device=device,
                no_cache=args.no_cache,
                limit=args.limit,
                description_dict=description_dict,
                decontamination_ngrams_path=args.decontamination_ngrams_path,
                check_integrity=args.check_integrity,
                write_out=args.write_out,
                output_base_path=args.output_base_path,
                tokenizer=model_path
            )
            time_dict = {}
            eval_time = time() - start_time
            time_dict['eval'] = eval_time
            print(f'eval took {eval_time} seconds')
            results['time'] = time_dict
            results['experiment_config'] = desc.__dict__

            results['ov_version'] = str(openvino.__version__)
            results_file = log_dir / 'results.json'
            print(results_file)
            all_results_paths.append(results_file.resolve())
            with results_file.open('w') as f:
                json.dump(results, f, indent=2)
            print(evaluator.make_table(results))

            shutil.rmtree(desc.model_path + "/model_cache")

        except Exception as error:
            print(traceback.print_exc())
            print(f"Eval of desc={desc} failed: {error}")
            continue

    for path in all_results_paths:
        print(path, '\n')
        with path.open() as f:
            j = json.load(f)
            r = j['results']
            print(json.dumps(r, indent=4))


if __name__ == "__main__":
    main()
