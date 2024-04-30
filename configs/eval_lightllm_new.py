from mmengine.config import read_base
from opencompass.models import LightllmAPI
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    # from .summarizers.leaderboard import summarizer
    from .datasets.humaneval.humaneval_gen_a82cae import humaneval_datasets
    from .datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from .datasets.mmlu.mmlu_gen_4d595a import mmlu_datasets  # noqa: F401, F403
    from .datasets.gpqa.gpqa_gen import gpqa_datasets
    from .datasets.lambada.lambada_gen import lambada_datasets
    from .datasets.math.math_gen import math_datasets
    from .datasets.IFEval.IFEval_gen import ifeval_datasets
    from .datasets.math401.math401_gen import math401_datasets
    from .datasets.z_bench.z_bench_gen import z_bench_datasets
    from .datasets.gpqa.gpqa_gen import gpqa_datasets
    from .datasets.strategyqa.strategyqa_gen import strategyqa_datasets

datasets = [*humaneval_datasets]
# datasets = [*gsm8k_datasets]
datasets = [*mmlu_datasets]
# datasets = [*lambada_datasets]
datasets = [*math_datasets]
datasets = [*ifeval_datasets]
# datasets = [*math401_datasets]
# datasets = [*z_bench_datasets]
# datasets = [*gpqa_datasets]
datasets = [*strategyqa_datasets]

'''
# Prompt template for InternLM2-Chat
# https://github.com/InternLM/InternLM/blob/main/chat/chat_format.md

_meta_template = dict(
    begin='<|im_start|>system\nYou are InternLM2-Chat, a harmless AI assistant<|im_end|>\n',
    round=[
        dict(role='HUMAN', begin='<|im_start|>user\n', end='<|im_end|>\n'),
        dict(role='BOT', begin='<|im_start|>assistant\n', end='<|im_end|>\n', generate=True),
    ]
)
'''

# _meta_template = dict(
#     begin='<|begin_of_text|><|start_header_id|>system<|end_header_id|>你是一个聪明的人工智能助手，特别擅长解答数学题目,对数学题目，解题步骤要详细，同时要对每个步骤进行校验，对最终结果进行检查，如果发现问题需要重新梳理思路再解答<|eot_id|>',
#     round=[
#         dict(role='HUMAN', begin='<|start_header_id|>user<|end_header_id|>', end='<|eot_id|>'),
#         dict(role='BOT', begin='<|start_header_id|>assistant<|end_header_id|>', end='<|eot_id|>', generate=True),
#     ]
# )

_meta_template = dict(
    begin='<|begin_of_text|><|start_header_id|>system<|end_header_id|>you are a smart assistant, 严格遵从指令进行问题回答<|eot_id|>',
    round=[
        dict(role='HUMAN', begin='<|start_header_id|>user<|end_header_id|>', end='<|eot_id|>'),
        dict(role='BOT', begin='<|start_header_id|>assistant<|end_header_id|>', end='<|eot_id|>', generate=True),
    ]
)


# _meta_template = None

topk_vlaue = 20
models = []
dest_ps = [0.0, 0.28, 0.38, 0.48, 0.78, 0.88, 0.98]
dest_ps = [0.0, 0.28, 0.48, 0.98]
for p in dest_ps:
    models.append(
         dict(
        abbr=f'LightllmAPI{p}_topk{topk_vlaue}',
        type=LightllmAPI,
        url='http://localhost:8012/generate',
        meta_template=_meta_template,
        batch_size=2,
        rate_per_worker=1111128000,
        retry=4,
        generation_kwargs=dict(
            do_sample=True,
            ignore_eos=False,
            max_new_tokens=1556,
            low_prob=p,
            top_k=topk_vlaue,
            stop_sequences = [" <|eot_id|>"]
            ),
    )
    )

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=80, strategy="split"),
    runner=dict(
        type=LocalRunner,
        max_num_workers=64,
        task=dict(type=OpenICLInferTask),
    ),
)
