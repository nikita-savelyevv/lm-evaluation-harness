from dataclasses import dataclass
from functools import partial
import gc
import shutil
from typing import Callable
import openvino.runtime as ov
from openvino import Core
import time
import queue
import atexit
import datetime
from nncf import compress_weights
from pathlib import Path
import threading
import matplotlib.pyplot as plt
from nncf.parameters import CompressWeightsMode
from tqdm import tqdm
import traceback
from optimum.intel import OVModelForCausalLM
from nncf import IgnoredScope
core = Core()


LOGS_DIR = Path("./logs_compress")



# from optimum.intel import OVModelForCausalLM
# MODEL_NAME = 'opt-125m'
# use_pkv = True
# ov_model = OVModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', use_cache=use_pkv, trust_remote_code=True, from_transformers=True)
# ov_model.save_pretrained('/home/nlyaly/projects/nncf/tests/openvino')
# ie = ov.Core()

@dataclass
class ExpDesc:
    model_id: str
    compress_fn: Callable
    exp_name: str
    is_bin_needed: bool = True
    def __str__(self):
        return f'{self.model_id}___{self.exp_name}'

int8_fn = compress_weights
nf4_fn = partial(compress_weights, mode=CompressWeightsMode.NF4, ratio=1, group_size=-1)
nf4_g128_fn = partial(compress_weights, mode=CompressWeightsMode.NF4, ratio=1, group_size=128)
mixed_g128_fn = partial(compress_weights, mode=CompressWeightsMode.NF4, ratio=0.5, group_size=128)
nf4_g64_fn = partial(compress_weights, mode=CompressWeightsMode.NF4, ratio=1, group_size=64)
nf4_g32_fn = partial(compress_weights, mode=CompressWeightsMode.NF4, ratio=1, group_size=32)
nf4_g128_r80_fn = partial(compress_weights, mode=CompressWeightsMode.NF4, ratio=0.8, group_size=128)
int4_g128_nozp_r80_fn = partial(compress_weights, mode=CompressWeightsMode.INT4_SYM, ratio=0.8, group_size=128)
int4_g128_nozp_fn = partial(compress_weights, mode=CompressWeightsMode.INT4_SYM, ratio=1, group_size=128)
int4_g64_nozp_fn = partial(compress_weights, mode=CompressWeightsMode.INT4_SYM, ratio=1, group_size=64)
int4_g128_fn = partial(compress_weights, mode=CompressWeightsMode.INT4_ASYM, ratio=1, group_size=128)


MODEL_IDS = [
    # 'facebook/opt-125m',
    'databricks/dolly-v2-3b',
    # 'openlm-research/open_llama_3b',
    'facebook/opt-6.7b',
    'bigscience/bloom-7b1',
    'togethercomputer/RedPajama-INCITE-7B-Instruct',
    # 'databricks/dolly-v2-12b',
    'meta-llama/Llama-2-7b-chat-hf',
    'meta-llama/Llama-2-13b-chat-hf',
    # 'chatglm2-6b',
    # 'chatglm-6b',
]

MODES_AND_NAMES = [
    # (nf4_g64_fn, 'nf4_ov_g64'),
    # (nf4_g128_fn, 'nf4_ov_g128'),
    # (int4_g128_fn, 'int4_g128'),
    # (int4_g128_nozp_fn, 'int4_g128_nozp'),
    # (int4_g64_nozp_fn, 'int4_g64_nozp'),
    # (int4_g128_nozp_r80_fn, 'int4_g128_nozp_r80'),
    # (nf4_g32_fn, 'nf4_ov_g32'),
    # (nf4_fn, 'nf4_ov'),
    (int8_fn, 'int8')
]


EXP_DESCS= [
#     # ExpDesc('meta-llama/Llama-2-13b-chat-hf', mixed_g128_fn, 'mixed_ov_g128'),
#     # ExpDesc('meta-llama/Llama-2-7b-chat-hf', mixed_g128_fn, 'mixed_ov_g128'),
#     # ExpDesc('meta-llama/Llama-2-13b-chat-hf', nf4_g128_fn, 'nf4_ov_g128', is_bin_needed=True),
#     # ExpDesc('meta-llama/Llama-2-7b-chat-hf', nf4_g128_fn, 'nf4_ov_g128', is_bin_needed=True),
#     ExpDesc('meta-llama/Llama-2-7b-chat-hf', nf4_g64_fn, 'nf4_ov_g64', is_bin_needed=True),
#     # ExpDesc('meta-llama/Llama-2-13b-chat-hf', nf4_g64_fn, 'nf4_g64_ov', is_bin_needed=True),
#     # ExpDesc('togethercomputer/RedPajama-INCITE-7B-Instruct', nf4_g64_fn, 'nf4_ov_g64', is_bin_needed=True),
#     # ExpDesc('chatglm2-6b', int8_fn, 'int8', is_bin_needed=True),
#     # ExpDesc('chatglm2-6b', nf4_fn, 'nf4_ov', is_bin_needed=True),
    # ExpDesc('bigscience/bloom-7b1', nf4_fn, 'nf4_ov', is_bin_needed=True),
    #     # ExpDesc('chatglm2-6b', nf4_g64_fn, 'nf4_ov_g64', is_bin_needed=True),
#     #ExpDesc('chatglm2-6b', nf4_g32_fn, 'nf4_ov_g32', is_bin_needed=True),
#     ExpDesc('dolly-v2-12b', nf4_g64_fn, 'nf4_ov_g64', is_bin_needed=True),
#     # ExpDesc('databricks/dolly-v2-3b', nf4_g64_fn, 'nf4_ov_g64', is_bin_needed=True),
#     # CLX

    # ExpDesc('bigscience/bloom-7b1', partial(compress_weights, mode=CompressWeightsMode.NF4, ratio=0.8, group_size=32), 'nf4_ov_g32_r80'),
    # ExpDesc('bigscience/bloom-7b1', partial(compress_weights, mode=CompressWeightsMode.NF4, ratio=0.6, group_size=64), 'nf4_ov_g64_r60'),
    # ExpDesc('bigscience/bloom-7b1', partial(compress_weights, mode=CompressWeightsMode.NF4, ratio=0.6, group_size=128), 'nf4_ov_g128_r60'),
    # ExpDesc('bigscience/bloom-7b1', partial(compress_weights, mode=CompressWeightsMode.INT4_ASYM, ratio=0.8, group_size=32), 'int4_ov_g32_r80'),
    # ExpDesc('bigscience/bloom-7b1', partial(compress_weights, mode=CompressWeightsMode.INT4_ASYM, ratio=0.6, group_size=64), 'int4_ov_g64_r60'),
    # ExpDesc('bigscience/bloom-7b1', partial(compress_weights, mode=CompressWeightsMode.INT4_ASYM, ratio=0.6, group_size=128), 'int4_ov_g128_r60'),

    # ExpDesc('togethercomputer/RedPajama-INCITE-7B-Instruct', partial(compress_weights, mode=CompressWeightsMode.NF4, ratio=0.8, group_size=32), 'nf4_ov_g32_r80'),
    # ExpDesc('togethercomputer/RedPajama-INCITE-7B-Instruct', partial(compress_weights, mode=CompressWeightsMode.NF4, ratio=0.6, group_size=32), 'nf4_ov_g32_r60'),


    # ExpDesc('databricks/dolly-v2-3b', partial(compress_weights, mode=CompressWeightsMode.NF4, ratio=0.8, group_size=128), 'nf4_ov_g128_r80'),
    # ExpDesc('databricks/dolly-v2-3b', partial(compress_weights, mode=CompressWeightsMode.NF4, ratio=0.6, group_size=128), 'nf4_ov_g128_r60'),

    # ExpDesc('meta-llama/Llama-2-7b-chat-hf', partial(compress_weights, mode=CompressWeightsMode.NF4, ratio=1, group_size=128), 'nf4_ov_g128'),
    # ExpDesc('meta-llama/Llama-2-13b-chat-hf', partial(compress_weights, mode=CompressWeightsMode.NF4, ratio=1, group_size=128), 'nf4_ov_g128'),

    # ExpDesc('facebook/opt-6.7b', partial(compress_weights, mode=CompressWeightsMode.NF4, ratio=1, group_size=128), 'nf4_ov_g128'),
    # ExpDesc('facebook/opt-6.7b', partial(compress_weights, mode=CompressWeightsMode.INT4_ASYM, ratio=1, group_size=64), 'int4_ov_g64'),
    # ExpDesc('facebook/opt-6.7b', partial(compress_weights, mode=CompressWeightsMode.INT4_ASYM, ratio=0.8, group_size=64), 'int4_ov_g64_r80'),
    # ExpDesc('facebook/opt-6.7b', partial(compress_weights, mode=CompressWeightsMode.INT4_ASYM, ratio=0.6, group_size=64), 'int4_ov_g64_r60'),
    # ExpDesc('facebook/opt-6.7b', partial(compress_weights, mode=CompressWeightsMode.INT4_ASYM, ratio=1, group_size=32), 'int4_ov_g32'),
    # ExpDesc('facebook/opt-6.7b', partial(compress_weights, mode=CompressWeightsMode.INT4_ASYM, ratio=0.8, group_size=32), 'int4_ov_g32_r80'),
    # ExpDesc('facebook/opt-6.7b', partial(compress_weights, mode=CompressWeightsMode.INT4_SYM, ratio=1, group_size=128), 'int4_g128'),
    # ExpDesc('databricks/dolly-v2-3b', partial(compress_weights, mode=CompressWeightsMode.INT4_ASYM, ratio=0.5, group_size=32), 'int4_ov_g32_r50'),
    # ExpDesc('databricks/dolly-v2-3b', partial(compress_weights, mode=CompressWeightsMode.INT4_ASYM, ratio=0.4, group_size=64), 'int4_ov_g64_r40'),

    # ExpDesc('togethercomputer/RedPajama-INCITE-7B-Instruct', partial(compress_weights, mode=CompressWeightsMode.INT4_ASYM, ratio=1, group_size=128), 'int4_ov_g128'),
    # ExpDesc('togethercomputer/RedPajama-INCITE-7B-Instruct', partial(compress_weights, mode=CompressWeightsMode.INT4_ASYM, ratio=0.8, group_size=128), 'int4_ov_g128_r80'),

    # ExpDesc('meta-llama/Llama-2-13b-chat-hf', partial(compress_weights, mode=CompressWeightsMode.INT4_SYM, ratio=0.8, group_size=64), 'int4_ov_g64_nozp_r80'),

#     # ExpDesc('open_llama_3b', nf4_g128_fn, 'nf4_ov_g128', is_bin_needed=True),
#     # ExpDesc('open_llama_3b', nf4_fn, 'nf4_ov', is_bin_needed=True),
    ExpDesc('HuggingFaceH4/zephyr-7b-beta', partial(compress_weights, mode=CompressWeightsMode.INT4_ASYM, ratio=1, group_size=128), 'int4_g128'),
    # ExpDesc('THUDM/chatglm3-6b', partial(compress_weights, mode=CompressWeightsMode.INT4_ASYM, ratio=1, group_size=128, ignored_scope=IgnoredScope(["__module.transformer.embedding.word_embeddings/aten::embedding/Gather"])), 'int4_g128'),
]

# EXP_DESCS = [ExpDesc(model_id, fn, name) for model_id in MODEL_IDS for fn, name in MODES_AND_NAMES]



is_bin_needed = True
cache_dir = Path('cache')
ov_name = 'openvino_model.xml'

# for model_id in MODEL_IDS:
#     for compress_fn, exp_name in MODES_AND_NAMES:
for desc in tqdm(EXP_DESCS):
    model_id = desc.model_id
    compress_fn = desc.compress_fn
    exp_name = desc.exp_name
    model_name = Path(model_id).name
    SRC_PATH = cache_dir / model_name / 'fp32'/  ov_name
    print(SRC_PATH)
    try:
        if not SRC_PATH.with_suffix('.bin').exists():
            use_pkv = True
            ov_model = OVModelForCausalLM.from_pretrained(model_id, use_cache=use_pkv, trust_remote_code=True, export=True)
            ov_model.save_pretrained(SRC_PATH.parent)
            ov_model._save_config(SRC_PATH.parent)
            fp32_model = ov_model.model
        else:
            fp32_model = core.read_model(model=SRC_PATH)
    except Exception as error:
        print("Reading FP32 model failed:", error)
        continue

    print(f'\n\n{model_id}___{exp_name}')

    DST_PATH = cache_dir / model_name / exp_name /  ov_name
    DST_PATH.parent.mkdir(exist_ok=True)
    print(DST_PATH)
    shutil.copyfile(SRC_PATH.parent / 'config.json', DST_PATH.parent / 'config.json')

    try:
        start = time.time()
        model = compress_fn(fp32_model)
        print(f'compressing weights took {(time.time() - start):.1f} seconds')
    except Exception as error:
        print("Compression failed:", error)
        print(traceback.print_exc())
        continue

    start = time.time()
    ov.save_model(model, DST_PATH, compress_to_fp16=False)
    print(f"saving model {DST_PATH} took {(time.time() - start):.1f} seconds")

    if not is_bin_needed:
        file_to_remove = DST_PATH.rename(DST_PATH.with_suffix('.bin'))
        Path.unlink(file_to_remove)

    del model
    gc.collect()