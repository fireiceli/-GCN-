import warnings

# from datasets import load_dataset, load_metric
# from typing import Optional,Tuple,Union
# import spacy
# import torch.nn as nn
# from torch.nn import CrossEntropyLoss
# import torch
from transformers import (
    AutoConfig,
    # AutoModelForSeq2SeqLM,
    AutoTokenizer,
    # DataCollatorForSeq2Seq,
    # HfArgumentParser,
    # MBartTokenizer,
    # default_data_collator,
    # set_seed,
    # T5ForConditionalGeneration,
    # PreTrainedModel, T5PreTrainedModel,
)
# from seq2seq.constrained_bart import gcn_model
# from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
# from transformers.models.t5.modeling_t5 import T5LayerNorm

# config1 = AutoConfig.from_pretrained(
#         'local_base',
#         cache_dir=None,
#         revision='main',
#         use_auth_token=None,
#     )
# tokenizer1 = AutoTokenizer.from_pretrained(
#         'local_base',
#         cache_dir=None,
#         use_fast=False,
#         revision='main',
#         use_auth_token=None,
#     )

config2 = AutoConfig.from_pretrained(
        'local_bart',
        cache_dir=None,
        revision='main',
        use_auth_token=None,
    )
config2.bos_token_id=None
tokenizer2 = AutoTokenizer.from_pretrained(
        'local_bart',
        cache_dir=None,
        use_fast=False,
        revision='main',
        use_auth_token=None,
    )
# print(tokenizer1.pad_token_id==0)
# print(tokenizer1.convert_tokens_to_ids(tokenizer1.unk_token))
s="Yet Putin has been far more reserved ."
print(tokenizer2(s))
inputs=tokenizer2(s)["input_ids"]
print(tokenizer2.batch_decode(inputs))
# print(tokenizer1.encode('Putin'))
# print(tokenizer1.encode('been'))
# print(tokenizer1.encode(s))
# print(tokenizer1.tokenize(s))
# print(config2.num_beams)

# model = AutoModelForSeq2SeqLM.from_pretrained(
#         't5-base',
#         from_tf=False,
#         config=config,
#         cache_dir=None,
#         revision='main',
#         use_auth_token=None,
#     )


# g_model=gcn_model(config)
# g_model.load_state_dict(torch.load('span_pretrained'))
# for name,parameters in model.named_parameters():
#     print(name,':',parameters.size())
# print(config.num_beams)





