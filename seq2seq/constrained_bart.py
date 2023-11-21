#!/usr/bin/env python
# -*- coding:utf-8 -*-
import logging
import math
import os
import sys

from torch import TensorType
from torch.nn import CrossEntropyLoss
#from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Union, List, Callable, Dict, Tuple, Any, Optional, Iterable
import numpy as np
from torch.cuda.amp import autocast
from transformers.generation_utils import GenerationMixin
from transformers.modeling_utils import ModuleUtilsMixin

from transformers.models.t5.modeling_t5 import T5LayerNorm

from transformers import (
    PreTrainedTokenizer,
    EvalPrediction,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq, PreTrainedModel, T5PreTrainedModel, LogitsProcessorList, StoppingCriteriaList,
    T5ForConditionalGeneration, T5Config,
)

from extraction.event_schema import EventSchema
from extraction.extract_constraint import get_constraint_decoder
from extraction.extraction_metrics import get_extract_metrics
from seq2seq.label_smoother_sum import SumLabelSmoother
from seq2seq.utils import lmap

from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from dataclasses import dataclass
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def add_logging_file(training_args):
    fh = logging.FileHandler(os.path.join(training_args.output_dir.rstrip(os.sep) + '.log'))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)


def decode_tree_str(sequences: Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor"],
                    tokenizer: PreTrainedTokenizer) -> List[str]:
    def clean_tree_text(x):
        return x.replace('<pad>', '').replace('<s>', '').replace('</s>', '').strip()

    sequences = np.where(sequences != -100, sequences, tokenizer.pad_token_id)

    str_list = tokenizer.batch_decode(sequences, skip_special_tokens=False)
    return lmap(clean_tree_text, str_list)


def build_compute_extract_metrics_event_fn(decoding_type_schema: EventSchema,
                                           decoding_format: str,
                                           tokenizer: PreTrainedTokenizer) -> Callable[[EvalPrediction], Dict]:
    def non_pad_len(tokens: np.ndarray) -> int:
        return np.count_nonzero(tokens != tokenizer.pad_token_id)

    def decode_pred(pred: EvalPrediction) -> Tuple[List[str], List[str]]:
        return decode_tree_str(pred.predictions, tokenizer), decode_tree_str(pred.label_ids, tokenizer)

    def extraction_metrics(pred: EvalPrediction) -> Dict:
        pred_str, label_str = decode_pred(pred)
        extraction = get_extract_metrics(pred_lns=pred_str, tgt_lns=label_str, label_constraint=decoding_type_schema,
                                         decoding_format=decoding_format)
        # rouge: Dict = calculate_rouge(pred_str, label_str)
        summ_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
        extraction.update({"gen_len": summ_len})
        # extraction.update( )
        return extraction

    compute_metrics_fn = extraction_metrics
    return compute_metrics_fn


@dataclass
class ConstraintSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    """
    Parameters:
        constraint_decoding (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use Constraint Decoding
        structure_weight (:obj:`float`, `optional`, defaults to :obj:`None`):
    """
    constraint_decoding: bool = field(default=False, metadata={"help": "Whether to Constraint Decoding or not."})
    label_smoothing_sum: bool = field(default=False,
                                      metadata={"help": "Whether to use sum token loss for label smoothing"})


class ConstraintSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, decoding_type_schema=None, decoding_format='tree', source_prefix=None, edge_exist=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.decoding_format = decoding_format
        self.decoding_type_schema = decoding_type_schema

        #########增加边的存在状态适应generate参数
        self.edge_exist=edge_exist
        ############################

        # Label smoothing by sum token loss, different from different Label smootheing
        if self.args.label_smoothing_sum and self.args.label_smoothing_factor != 0:
            self.label_smoother = SumLabelSmoother(epsilon=self.args.label_smoothing_factor)
            print('Using %s' % self.label_smoother)
        elif self.args.label_smoothing_factor != 0:
            print('Using %s' % self.label_smoother)
        else:
            self.label_smoother = None

        if self.args.constraint_decoding:
            self.constraint_decoder = get_constraint_decoder(tokenizer=self.tokenizer,
                                                             type_schema=self.decoding_type_schema,
                                                             decoding_schema=self.decoding_format,
                                                             source_prefix=source_prefix)
        else:
            self.constraint_decoder = None

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        def prefix_allowed_tokens_fn(batch_id, sent):
            # print(self.tokenizer.convert_ids_to_tokens(inputs['labels'][batch_id]))
            src_sentence = inputs['input_ids'][batch_id]
            return self.constraint_decoder.constraint_decoding(src_sentence=src_sentence,
                                                               tgt_generated=sent)

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model=model,
                inputs=inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn if self.constraint_decoder else None,
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        if(self.edge_exist==True):
            gen_kwargs = {
                "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
                "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
                "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn if self.constraint_decoder else None,
                # "length_penalty": 0,
                "edges": inputs['edges'],
            }
        else:
            gen_kwargs = {
                "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
                "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
                "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn if self.constraint_decoder else None,
            }

        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )



        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None


        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])


        if self.args.prediction_loss_only:
            return loss, None, None

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return loss, generated_tokens, labels

#########################################################################################
#####################以下是添加部分########################################################




def get_adj(maxlen,edges):
    res=[[0 for _ in range(maxlen)] for _ in range(maxlen)]
    n=len(edges[0])
    for i in range(n):
        try:
            res[edges[0][i]][edges[1][i]] += 1
        except:
            logger.info('详细信息如下：')
            logger.info('maxlen={}'.format(maxlen))
            logger.info('edges=')
            logger.info(edges)


    """
    对称归一化邻接矩阵
    :param adjacency: 邻接矩阵，二维 numpy 数组，shape 为 (num_nodes, num_nodes)
    :return: 归一化后的邻接矩阵，二维 numpy 数组，shape 为 (num_nodes, num_nodes)
    """
    np.seterr(divide='ignore', invalid='ignore')  # 消除被除数为0的警告
    # 计算度矩阵
    adjacency=np.array(res)
    adjacency = adjacency + np.eye(adjacency.shape[0])
    degree = np.sum(adjacency, axis=1)
    # 计算度矩阵的逆矩阵
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.
    # 对称归一化邻接矩阵
    normalized_adjacency = np.multiply(np.multiply(degree_inv_sqrt.reshape(-1, 1), adjacency), degree_inv_sqrt.reshape(1, -1))
    return normalized_adjacency.tolist()

@dataclass
class DataCollatorForSeq2Seq_edges():
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, ] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        inputs = [feature['input_ids'] for feature in features]                 # 对边进行处理,转化为拉普拉斯矩阵
        max_length = max(len(l) for l in inputs)
        for feature in features:
            feature['edges'] = get_adj(max_length,feature['edges'])



        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features


class encoder(nn.Module):
    def __init__(self,e,s,c):
        super().__init__()
        self.encod=e
        self.s=s
        self.conv=c

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            edges=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            encoder_head_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        assert edges != None, '邻接矩阵未输入'

        inputs_embeds=self.s(input_ids)
        inputs_embeds=self.conv(inputs_embeds,edges)

        # return self.encod(input_ids=None,
        # attention_mask=attention_mask,
        # encoder_hidden_states=encoder_hidden_states,
        # encoder_attention_mask=encoder_attention_mask,
        # inputs_embeds=inputs_embeds,
        # head_mask=head_mask,
        # encoder_head_mask=encoder_head_mask,
        # past_key_values=past_key_values,
        # use_cache=use_cache,
        # output_attentions=output_attentions,
        # output_hidden_states=output_hidden_states,
        # return_dict=return_dict,)

        return self.encod(input_ids=None,
        attention_mask=attention_mask,
        head_mask=None,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,)



class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.liner1=nn.Linear(in_features,out_features,bias=True)
        self.liner2=nn.Linear(in_features,out_features,bias=True)
        self.layer_norm = T5LayerNorm(out_features)
        self.dropout=nn.Dropout(0.1)


    def forward(self, x, adj):
        #计算图卷积
        y = self.dropout(x)
        y = F.relu(self.liner1(y))
        y = self.layer_norm(torch.matmul(adj, y))
        y = self.liner2(self.dropout(y))
        y = self.layer_norm(torch.matmul(adj, y))
        return x+y




#不使用pretrainmodel
# state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME))
# self.model.load_state_dict(state_dict)

class gcn_model(nn.Module, ModuleUtilsMixin, GenerationMixin):
    def __init__(self,config,model):
        super().__init__()
        self.model=model
        self.config=config
        self.conv=GCNLayer(self.config.d_model,self.config.d_model)

    def get_encoder(self):
        return encoder(self.model.model.encoder,self.model.model.shared,self.conv)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "edges": kwargs['edges'],
        }

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> torch.nn.Embedding:
        return self.model.resize_token_embeddings(new_num_tokens)

    # @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self,
        input_ids=None,
        attention_mask=None,
        edges=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,) :

        assert edges != None, '邻接矩阵未输入'
        if (input_ids != None):
            inputs_embeds=self.model.model.shared(input_ids)    #bart模型embeding不在外层
            inputs_embeds=self.conv(inputs_embeds,edges)
            input_ids=None
        return self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            decoder_input_ids=decoder_input_ids,
                            decoder_attention_mask=decoder_attention_mask,
                            head_mask=head_mask,
                            decoder_head_mask=decoder_head_mask,
                            encoder_outputs=encoder_outputs,
                            past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds,
                            decoder_inputs_embeds=decoder_inputs_embeds,
                            labels=labels,
                            use_cache=use_cache,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict,)

def main(): pass


if __name__ == "__main__":
    main()
