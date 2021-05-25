# coding=utf-8
"""PyTorch BERT model."""
import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil
import numpy as np
from scipy.stats import truncnorm

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from vteacher.file_utils import cached_path
# from .loss import LabelSmoothingLoss
# from .trace import get_best_sequence

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 relax_projection=0,
                 new_pos_ids=False,
                 initializer_range=0.02,
                 task_idx=None,
                 fp32_embedding=False,
                 ffn_type=0,
                 label_smoothing=None,
                 num_qkv=0,
                 seg_emb=True):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.relax_projection = relax_projection
            self.new_pos_ids = new_pos_ids
            self.initializer_range = initializer_range
            self.task_idx = task_idx
            self.fp32_embedding = fp32_embedding
            self.ffn_type = ffn_type
            self.label_smoothing = label_smoothing
            self.num_qkv = num_qkv
            self.seg_emb = seg_emb
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


# try:
#     from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
# except ImportError:
# print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


        
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            6, config.hidden_size)
        if hasattr(config, 'fp32_embedding'):
            self.fp32_embedding = config.fp32_embedding
        else:
            self.fp32_embedding = False

        if hasattr(config, 'new_pos_ids') and config.new_pos_ids:
            self.num_pos_emb = 4
        else:
            self.num_pos_emb = 1
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size*self.num_pos_emb)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, task_idx=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        if self.num_pos_emb > 1:
            num_batch = position_embeddings.size(0)
            num_pos = position_embeddings.size(1)
            position_embeddings = position_embeddings.view(
                num_batch, num_pos, self.num_pos_emb, -1)[torch.arange(0, num_batch).long(), :, task_idx, :]

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        if self.fp32_embedding:
            embeddings = embeddings.half()
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if hasattr(config, 'num_qkv') and (config.num_qkv > 1):
            self.num_qkv = config.num_qkv
        else:
            self.num_qkv = 1

        self.query = nn.Linear(
            config.hidden_size, self.all_head_size*self.num_qkv)
        self.key = nn.Linear(config.hidden_size,
                             self.all_head_size*self.num_qkv)
        self.value = nn.Linear(
            config.hidden_size, self.all_head_size*self.num_qkv)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.uni_debug_flag = True if os.getenv(
            'UNI_DEBUG_FLAG', '') else False
        if self.uni_debug_flag:
            self.register_buffer('debug_attention_probs',
                                 torch.zeros((512, 512)))
        if hasattr(config, 'seg_emb') and config.seg_emb:
            self.b_q_s = nn.Parameter(torch.zeros(
                1, self.num_attention_heads, 1, self.attention_head_size))
            self.seg_emb = nn.Embedding(
                6, self.all_head_size)
        else:
            self.b_q_s = None
            self.seg_emb = None

    def transpose_for_scores(self, x, mask_qkv=None):
        if self.num_qkv > 1:
            sz = x.size()[:-1] + (self.num_qkv,
                                  self.num_attention_heads, self.all_head_size)
            # (batch, pos, num_qkv, head, head_hid)
            x = x.view(*sz)
            if mask_qkv is None:
                x = x[:, :, 0, :, :]
            elif isinstance(mask_qkv, int):
                x = x[:, :, mask_qkv, :, :]
            else:
                # mask_qkv: (batch, pos)
                if mask_qkv.size(1) > sz[1]:
                    mask_qkv = mask_qkv[:, :sz[1]]
                # -> x: (batch, pos, head, head_hid)
                x = x.gather(2, mask_qkv.view(sz[0], sz[1], 1, 1, 1).expand(
                    sz[0], sz[1], 1, sz[3], sz[4])).squeeze(2)
        else:
            sz = x.size()[:-1] + (self.num_attention_heads,
                                  self.attention_head_size)
            # (batch, pos, head, head_hid)
            x = x.view(*sz)
        # (batch, head, pos, head_hid)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, history_states=None, mask_qkv=None, seg_ids=None, att_ids=None):
        if history_states is None:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
        else:
            x_states = torch.cat((history_states, hidden_states), dim=1)
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)

        query_layer = self.transpose_for_scores(mixed_query_layer, mask_qkv)
        key_layer = self.transpose_for_scores(mixed_key_layer, mask_qkv)
        value_layer = self.transpose_for_scores(mixed_value_layer, mask_qkv)
#         print(mixed_query_layer.size(), mixed_key_layer.size(), mixed_value_layer.size())
#         print(query_layer.size(), key_layer.size(), value_layer.size())
        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch, head, pos, pos)
        attention_scores = torch.matmul(
            query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))
#         print(query_layer.size(), key_layer.size(), attention_scores.size(), attention_mask.size())
        if self.seg_emb is not None:
            seg_rep = self.seg_emb(seg_ids)
            # (batch, pos, head, head_hid)
            seg_rep = seg_rep.view(seg_rep.size(0), seg_rep.size(
                1), self.num_attention_heads, self.attention_head_size)
            qs = torch.einsum('bnih,bjnh->bnij',
                              query_layer+self.b_q_s, seg_rep)
            attention_scores = attention_scores + qs

        # attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores_1 = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores_1)
#         attention_probs = self.dropout(attention_probs)
            # Normalize the attention scores to probabilities.
            
        if self.uni_debug_flag:
            _pos = attention_probs.size(-1)
            self.debug_attention_probs[:_pos, :_pos].copy_(
                attention_probs[0].mean(0).view(_pos, _pos))

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        
        if att_ids is not None:
            cand1 = []
            cand2 = []
            cand3 = []
            cand1_r = []
            cand2_r = []
            cand3_r = []
            real_mask = torch.tensor((attention_mask == 0).detach().cpu().numpy()).cuda()
#             attention_probs_att = F.softmax(attention_scores, dim=-1)
            for i, att_id in enumerate(att_ids): 
#                 print(att_id, attention_scores[i].size())
#                 real_mask[i][:, :(att_id[0]+1).int(), 0:1]
                cand1i = attention_probs[i][:, :(att_id[0]).int(), (att_id[0]).int():(att_id[1]).int()] * real_mask[i][:, :(att_id[0]).int(), 0:1]
#                 print(attention_scores[i].max(), attention_scores[i][:, :(att_id[0]+1).int()].max(), attention_scores[i][:, :(att_id[0]+1).int(), (att_id[0]+1).int():(att_id[1]+1).int()].max(), cand1i.max())
#                 cand_mask = cand1i!=0
                cand1i_max = torch.max(cand1i, 1)[0].mean(1)
#     /att_id[0]/(att_id[1]-att_id[0])
#                 cand1i = (cand1i * cand_mask).sum(-1, keepdim=True)/cand_mask.sum(-1, keepdim=True)
#                 cand1i = (cand1i * cand_mask).sum(1)/cand_mask.sum(1)
            
                cand1i_r = attention_probs[i][:, (att_id[0]).int():(att_id[1]).int(), :(att_id[0]).int()] * real_mask[i][:, 0:1, :(att_id[0]).int()]
#                 cand_mask = cand1i_r!=0
                cand1i_r_max = torch.max(cand1i_r, 1)[0].mean(1)
#     /att_id[0]/(att_id[1]-att_id[0])
#                 cand1i_r = (cand1i_r * cand_mask).sum(-1, keepdim=True)/cand_mask.sum(-1, keepdim=True)
#                 cand1i_r = (cand1i_r * cand_mask).sum(1)/cand_mask.sum(1)
            
                cand1.append(cand1i_max)
                cand1_r.append(cand1i_r_max)
                
                cand2i = attention_probs[i][:, :(att_id[0]).int(), (att_id[1]).int():(att_id[2]).int()] * real_mask[i][:, :(att_id[0]).int(), 0:1]
#                 cand_mask = cand2i!=0
                cand2i_max = torch.max(cand2i, 1)[0].mean(1)
#     /att_id[0]/(att_id[2]-att_id[1])
#                 cand2i = (cand2i * cand_mask).sum(-1, keepdim=True)/cand_mask.sum(-1, keepdim=True)
#                 cand2i = (cand2i * cand_mask).sum(1)/cand_mask.sum(1)                
                
                cand2i_r = attention_probs[i][:, (att_id[1]).int():(att_id[2]).int(), :(att_id[0]).int()] * real_mask[i][:, 0:1, :(att_id[0]).int()]
#                 cand_mask = cand2i_r!=0
                cand2i_r_max = torch.max(cand2i_r, 1)[0].mean(1)
#     /att_id[0]/(att_id[2]-att_id[1])
#                 cand2i_r = (cand2i_r * cand_mask).sum(-1, keepdim=True)/cand_mask.sum(-1, keepdim=True)
#                 cand2i_r = (cand2i_r * cand_mask).sum(1)/cand_mask.sum(1)
                
                cand2.append(cand2i_max)
                cand2_r.append(cand2i_r_max)
                
                cand3i = attention_probs[i][:, :(att_id[0]).int(), (att_id[2]).int():(att_id[3]).int()] * real_mask[i][:, :(att_id[0]).int(), 0:1]
#                 cand_mask = cand3i!=0
                cand3i_max = torch.max(cand3i, 1)[0].mean(1)
#     /att_id[0]/(att_id[3]-att_id[2])
#                 cand3i = (cand3i * cand_mask).sum(-1, keepdim=True)/cand_mask.sum(-1, keepdim=True)
#                 cand3i = (cand3i * cand_mask).sum(1)/cand_mask.sum(1)
                
                cand3i_r = attention_probs[i][:, (att_id[2]).int():(att_id[3]).int(), :(att_id[0]).int()] * real_mask[i][:, 0:1, :(att_id[0]).int()]
#                 cand_mask = cand3i_r!=0
                cand3i_r_max = torch.max(cand3i_r, 1)[0].mean(1)
#     /att_id[0]/(att_id[3]-att_id[2])
#                 cand3i_r = (cand3i_r * cand_mask).sum(-1, keepdim=True)/cand_mask.sum(-1, keepdim=True)
#                 cand3i_r = (cand3i_r * cand_mask).sum(1)/cand_mask.sum(1)
                
                cand3.append(cand3i_max)
                cand3_r.append(cand3i_r_max)
            
            selective_att_scores = torch.cat([torch.stack(cand1, 0).unsqueeze(-1), torch.stack(cand2, 0).unsqueeze(-1), torch.stack(cand3, 0).unsqueeze(-1)], -1)
#             print(selective_att_scores.size(), torch.stack(cand1, 0).unsqueeze(1).size(), torch.stack(cand1, 0).size())
            selective_att_scores_r = torch.cat([torch.stack(cand1_r, 0).unsqueeze(-1), torch.stack(cand2_r, 0).unsqueeze(-1), torch.stack(cand3_r, 0).unsqueeze(-1)], -1)
#             print(selective_att_scores.size(), selective_att_scores_r.size())
#             print(selective_att_scores[0], selective_att_scores_r[0])
#             selective_att_scores = nn.Softmax(dim=-1)(selective_att_scores)
        else:
            selective_att_scores = 0
            selective_att_scores_r = 0
            
        context_layer = torch.matmul(attention_probs, value_layer)
#         print(value_layer.size(), attention_probs.size(), context_layer.size())
        context_layer_1 = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer_1.size()[
            :-2] + (self.all_head_size,)
        context_layer_2 = context_layer_1.view(*new_context_layer_shape)
#         print(context_layer_2.size())
            
        return context_layer_2, selective_att_scores, selective_att_scores_r, attention_probs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, history_states=None, mask_qkv=None, seg_ids=None, att_ids=None):
        self_output, selective_att_scores, selective_att_scores_r, attention_probs = self.self(
            input_tensor, attention_mask, history_states=history_states, mask_qkv=mask_qkv, seg_ids=seg_ids, att_ids=att_ids)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, selective_att_scores, selective_att_scores_r, attention_probs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TransformerFFN(nn.Module):
    def __init__(self, config):
        super(TransformerFFN, self).__init__()
        self.ffn_type = config.ffn_type
        assert self.ffn_type in (1, 2)
        if self.ffn_type in (1, 2):
            self.wx0 = nn.Linear(config.hidden_size, config.hidden_size)
        if self.ffn_type in (2,):
            self.wx1 = nn.Linear(config.hidden_size, config.hidden_size)
        if self.ffn_type in (1, 2):
            self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        if self.ffn_type in (1, 2):
            x0 = self.wx0(x)
            if self.ffn_type == 1:
                x1 = x
            elif self.ffn_type == 2:
                x1 = self.wx1(x)
            out = self.output(x0 * x1)
        out = self.dropout(out)
        out = self.LayerNorm(out + x)
        return out


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.ffn_type = config.ffn_type
        if self.ffn_type:
            self.ffn = TransformerFFN(config)
        else:
            self.intermediate = BertIntermediate(config)
            self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, history_states=None, mask_qkv=None, seg_ids=None, att_ids=None):
        attention_output, selective_att_scores, selective_att_scores_r, attention_probs = self.attention(
            hidden_states, attention_mask, history_states=history_states, mask_qkv=mask_qkv, seg_ids=seg_ids, att_ids=att_ids)
        if self.ffn_type:
            layer_output = self.ffn(attention_output)
        else:
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
        return layer_output, selective_att_scores, selective_att_scores_r, attention_probs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
#         self.layer = nn.ModuleList([copy.deepcopy(layer)
#                                     for _ in range(2)])

        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states=None, attention_mask=None, output_all_encoded_layers=True, prev_embedding=None, prev_encoded_layers=None, mask_qkv=None, seg_ids=None, att_ids=None):
        # history embedding and encoded layer must be simultanously given
        assert (prev_embedding is None) == (prev_encoded_layers is None)
        sum_selective_att_scores = []
        sum_selective_att_scores_r = []
        all_encoder_layers = []
        all_attention_maps = []
        if (prev_embedding is not None) and (prev_encoded_layers is not None):
            history_states = prev_embedding
            for i, layer_module in enumerate(self.layer):
                hidden_states, selective_att_scores, selective_att_scores_r, attention_probs = layer_module(
                    hidden_states, attention_mask, history_states=history_states, mask_qkv=mask_qkv, seg_ids=seg_ids, att_ids=att_ids)
#                 print(selective_att_scores.size())
                sum_selective_att_scores += [selective_att_scores]
                sum_selective_att_scores_r += [selective_att_scores_r]
                all_attention_maps += [attention_probs]
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
                if prev_encoded_layers is not None:
                    history_states = prev_encoded_layers[i]
        else:
            for layer_module in self.layer:
                hidden_states, selective_att_scores, selective_att_scores_r, attention_probs = layer_module(
                    hidden_states, attention_mask, mask_qkv=mask_qkv, seg_ids=seg_ids, att_ids=att_ids)
#                 print(selective_att_scores.size())
                sum_selective_att_scores += [selective_att_scores]
                sum_selective_att_scores_r += [selective_att_scores_r]
                all_attention_maps += [attention_probs]
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
#         print(sum_selective_att_scores.size())
        
        return all_encoder_layers, sum_selective_att_scores, sum_selective_att_scores_r, all_attention_maps
           

class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        hid_size = config.hidden_size
        if hasattr(config, 'relax_projection') and (config.relax_projection > 1):
            hid_size *= config.relax_projection
        self.dense = nn.Linear(config.hidden_size, hid_size)
        self.LayerNorm = BertLayerNorm(hid_size, eps=1e-5)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(
            bert_model_embedding_weights.size(0)))
        if hasattr(config, 'relax_projection') and (config.relax_projection > 1):
            self.relax_projection = config.relax_projection
        else:
            self.relax_projection = 0
        self.fp32_embedding = config.fp32_embedding

        def convert_to_type(tensor):
            if self.fp32_embedding:
                return tensor.half()
            else:
                return tensor
        self.type_converter = convert_to_type
        self.converted = False

    def forward(self, hidden_states, task_idx=None):
        if not self.converted:
            self.converted = True
            if self.fp32_embedding:
                self.transform.half()
        hidden_states = self.transform(self.type_converter(hidden_states))
        if self.relax_projection > 1:
            num_batch = hidden_states.size(0)
            num_pos = hidden_states.size(1)
            # (batch, num_pos, relax_projection*hid) -> (batch, num_pos, relax_projection, hid) -> (batch, num_pos, hid)
            hidden_states = hidden_states.view(
                num_batch, num_pos, self.relax_projection, -1)[torch.arange(0, num_batch).long(), :, task_idx, :]
        if self.fp32_embedding:
            hidden_states = F.linear(self.type_converter(hidden_states), self.type_converter(
                self.decoder.weight), self.type_converter(self.bias))
        else:
            hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(
            config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights, num_labels=2):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(
            config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, num_labels)

    def forward(self, sequence_output, pooled_output, task_idx=None):
        prediction_scores = self.predictions(sequence_output, task_idx)
        if pooled_output is None:
            seq_relationship_score = None
        else:
            seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class PreTrainedBertModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedBertModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name, state_dict=None, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-base-multilingual`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        if pretrained_model_name in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name]
        else:
            archive_file = pretrained_model_name
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(
                archive_file, cache_dir=cache_dir)
        except FileNotFoundError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name,
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file):
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        if ('config_path' in kwargs) and kwargs['config_path']:
            config_file = kwargs['config_path']
        else:
            config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)

        # define new type_vocab_size (there might be different numbers of segment ids)
        if 'type_vocab_size' in kwargs:
            config.type_vocab_size = kwargs['type_vocab_size']
        # define new relax_projection
        if ('relax_projection' in kwargs) and kwargs['relax_projection']:
            config.relax_projection = kwargs['relax_projection']
        # new position embedding
        if ('new_pos_ids' in kwargs) and kwargs['new_pos_ids']:
            config.new_pos_ids = kwargs['new_pos_ids']
        # define new relax_projection
        if ('task_idx' in kwargs) and kwargs['task_idx']:
            config.task_idx = kwargs['task_idx']
        # define new max position embedding for length expansion
        if ('max_position_embeddings' in kwargs) and kwargs['max_position_embeddings']:
            config.max_position_embeddings = kwargs['max_position_embeddings']
        # use fp32 for embeddings
        if ('fp32_embedding' in kwargs) and kwargs['fp32_embedding']:
            config.fp32_embedding = kwargs['fp32_embedding']
        # type of FFN in transformer blocks
        if ('ffn_type' in kwargs) and kwargs['ffn_type']:
            config.ffn_type = kwargs['ffn_type']
        # label smoothing
        if ('label_smoothing' in kwargs) and kwargs['label_smoothing']:
            config.label_smoothing = kwargs['label_smoothing']
        # dropout
        if ('hidden_dropout_prob' in kwargs) and kwargs['hidden_dropout_prob']:
            config.hidden_dropout_prob = kwargs['hidden_dropout_prob']
        if ('attention_probs_dropout_prob' in kwargs) and kwargs['attention_probs_dropout_prob']:
            config.attention_probs_dropout_prob = kwargs['attention_probs_dropout_prob']
        # different QKV
        if ('num_qkv' in kwargs) and kwargs['num_qkv']:
            config.num_qkv = kwargs['num_qkv']
        # segment embedding for self-attention
        if ('seg_emb' in kwargs) and kwargs['seg_emb']:
            config.seg_emb = kwargs['seg_emb']
        # initialize word embeddings
        _word_emb_map = None
        if ('word_emb_map' in kwargs) and kwargs['word_emb_map']:
            _word_emb_map = kwargs['word_emb_map']

        logger.info("Model config {}".format(config))

        # clean the arguments in kwargs
        for arg_clean in ('config_path', 'type_vocab_size', 'relax_projection', 'new_pos_ids', 'task_idx', 'max_position_embeddings', 'fp32_embedding', 'ffn_type', 'label_smoothing', 'hidden_dropout_prob', 'attention_probs_dropout_prob', 'num_qkv', 'seg_emb', 'word_emb_map'):
            if arg_clean in kwargs:
                del kwargs[arg_clean]

        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path)

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # initialize new segment embeddings
        _k = 'bert.embeddings.token_type_embeddings.weight'
        if (_k in state_dict) and (config.type_vocab_size != state_dict[_k].shape[0]):
            logger.info("config.type_vocab_size != state_dict[bert.embeddings.token_type_embeddings.weight] ({0} != {1})".format(
                config.type_vocab_size, state_dict[_k].shape[0]))
            if config.type_vocab_size > state_dict[_k].shape[0]:
                # state_dict[_k].data = state_dict[_k].data.resize_(config.type_vocab_size, state_dict[_k].shape[1])
                state_dict[_k].resize_(
                    config.type_vocab_size, state_dict[_k].shape[1])
                # L2R
                if config.type_vocab_size >= 3:
                    state_dict[_k].data[2, :].copy_(state_dict[_k].data[0, :])
                # R2L
                if config.type_vocab_size >= 4:
                    state_dict[_k].data[3, :].copy_(state_dict[_k].data[0, :])
                # S2S
                if config.type_vocab_size >= 6:
                    state_dict[_k].data[4, :].copy_(state_dict[_k].data[0, :])
                    state_dict[_k].data[5, :].copy_(state_dict[_k].data[1, :])
                if config.type_vocab_size >= 7:
                    state_dict[_k].data[6, :].copy_(state_dict[_k].data[1, :])
            elif config.type_vocab_size < state_dict[_k].shape[0]:
                state_dict[_k].data = state_dict[_k].data[:config.type_vocab_size, :]

        _k = 'bert.embeddings.position_embeddings.weight'
        n_config_pos_emb = 4 if config.new_pos_ids else 1
        if (_k in state_dict) and (n_config_pos_emb*config.hidden_size != state_dict[_k].shape[1]):
            logger.info("n_config_pos_emb*config.hidden_size != state_dict[bert.embeddings.position_embeddings.weight] ({0}*{1} != {2})".format(
                n_config_pos_emb, config.hidden_size, state_dict[_k].shape[1]))
            assert state_dict[_k].shape[1] % config.hidden_size == 0
            n_state_pos_emb = int(state_dict[_k].shape[1]/config.hidden_size)
            assert (n_state_pos_emb == 1) != (n_config_pos_emb ==
                                              1), "!!!!n_state_pos_emb == 1 xor n_config_pos_emb == 1!!!!"
            if n_state_pos_emb == 1:
                state_dict[_k].data = state_dict[_k].data.unsqueeze(1).repeat(
                    1, n_config_pos_emb, 1).reshape((config.max_position_embeddings, n_config_pos_emb*config.hidden_size))
            elif n_config_pos_emb == 1:
                if hasattr(config, 'task_idx') and (config.task_idx is not None) and (0 <= config.task_idx <= 3):
                    _task_idx = config.task_idx
                else:
                    _task_idx = 0
                state_dict[_k].data = state_dict[_k].data.view(
                    config.max_position_embeddings, n_state_pos_emb, config.hidden_size).select(1, _task_idx)

        # initialize new position embeddings
        _k = 'bert.embeddings.position_embeddings.weight'
        if _k in state_dict and config.max_position_embeddings != state_dict[_k].shape[0]:
            logger.info("config.max_position_embeddings != state_dict[bert.embeddings.position_embeddings.weight] ({0} - {1})".format(
                config.max_position_embeddings, state_dict[_k].shape[0]))
            if config.max_position_embeddings > state_dict[_k].shape[0]:
                old_size = state_dict[_k].shape[0]
                # state_dict[_k].data = state_dict[_k].data.resize_(config.max_position_embeddings, state_dict[_k].shape[1])
                state_dict[_k].resize_(
                    config.max_position_embeddings, state_dict[_k].shape[1])
                start = old_size
                while start < config.max_position_embeddings:
                    chunk_size = min(
                        old_size, config.max_position_embeddings - start)
                    state_dict[_k].data[start:start+chunk_size,
                                        :].copy_(state_dict[_k].data[:chunk_size, :])
                    start += chunk_size
            elif config.max_position_embeddings < state_dict[_k].shape[0]:
                state_dict[_k].data = state_dict[_k].data[:config.max_position_embeddings, :]

        # initialize relax projection
        _k = 'cls.predictions.transform.dense.weight'
        n_config_relax = 1 if (config.relax_projection <
                               1) else config.relax_projection
        if (_k in state_dict) and (n_config_relax*config.hidden_size != state_dict[_k].shape[0]):
            logger.info("n_config_relax*config.hidden_size != state_dict[cls.predictions.transform.dense.weight] ({0}*{1} != {2})".format(
                n_config_relax, config.hidden_size, state_dict[_k].shape[0]))
            assert state_dict[_k].shape[0] % config.hidden_size == 0
            n_state_relax = int(state_dict[_k].shape[0]/config.hidden_size)
            assert (n_state_relax == 1) != (n_config_relax ==
                                            1), "!!!!n_state_relax == 1 xor n_config_relax == 1!!!!"
            if n_state_relax == 1:
                _k = 'cls.predictions.transform.dense.weight'
                state_dict[_k].data = state_dict[_k].data.unsqueeze(0).repeat(
                    n_config_relax, 1, 1).reshape((n_config_relax*config.hidden_size, config.hidden_size))
                for _k in ('cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias'):
                    state_dict[_k].data = state_dict[_k].data.unsqueeze(
                        0).repeat(n_config_relax, 1).view(-1)
            elif n_config_relax == 1:
                if hasattr(config, 'task_idx') and (config.task_idx is not None) and (0 <= config.task_idx <= 3):
                    _task_idx = config.task_idx
                else:
                    _task_idx = 0
                _k = 'cls.predictions.transform.dense.weight'
                state_dict[_k].data = state_dict[_k].data.view(
                    n_state_relax, config.hidden_size, config.hidden_size).select(0, _task_idx)
                for _k in ('cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias'):
                    state_dict[_k].data = state_dict[_k].data.view(
                        n_state_relax, config.hidden_size).select(0, _task_idx)

        # initialize QKV
        _all_head_size = config.num_attention_heads * \
            int(config.hidden_size / config.num_attention_heads)
        n_config_num_qkv = 1 if (config.num_qkv < 1) else config.num_qkv
        for qkv_name in ('query', 'key', 'value'):
            _k = 'bert.encoder.layer.0.attention.self.{0}.weight'.format(
                qkv_name)
            if (_k in state_dict) and (n_config_num_qkv*_all_head_size != state_dict[_k].shape[0]):
                logger.info("n_config_num_qkv*_all_head_size != state_dict[_k] ({0}*{1} != {2})".format(
                    n_config_num_qkv, _all_head_size, state_dict[_k].shape[0]))
                for layer_idx in range(config.num_hidden_layers):
                    _k = 'bert.encoder.layer.{0}.attention.self.{1}.weight'.format(
                        layer_idx, qkv_name)
                    assert state_dict[_k].shape[0] % _all_head_size == 0
                    n_state_qkv = int(state_dict[_k].shape[0]/_all_head_size)
                    assert (n_state_qkv == 1) != (n_config_num_qkv ==
                                                  1), "!!!!n_state_qkv == 1 xor n_config_num_qkv == 1!!!!"
                    if n_state_qkv == 1:
                        _k = 'bert.encoder.layer.{0}.attention.self.{1}.weight'.format(
                            layer_idx, qkv_name)
                        state_dict[_k].data = state_dict[_k].data.unsqueeze(0).repeat(
                            n_config_num_qkv, 1, 1).reshape((n_config_num_qkv*_all_head_size, _all_head_size))
                        _k = 'bert.encoder.layer.{0}.attention.self.{1}.bias'.format(
                            layer_idx, qkv_name)
                        state_dict[_k].data = state_dict[_k].data.unsqueeze(
                            0).repeat(n_config_num_qkv, 1).view(-1)
                    elif n_config_num_qkv == 1:
                        if hasattr(config, 'task_idx') and (config.task_idx is not None) and (0 <= config.task_idx <= 3):
                            _task_idx = config.task_idx
                        else:
                            _task_idx = 0
                        assert _task_idx != 3, "[INVALID] _task_idx=3: n_config_num_qkv=1 (should be 2)"
                        if _task_idx == 0:
                            _qkv_idx = 0
                        else:
                            _qkv_idx = 1
                        _k = 'bert.encoder.layer.{0}.attention.self.{1}.weight'.format(
                            layer_idx, qkv_name)
                        state_dict[_k].data = state_dict[_k].data.view(
                            n_state_qkv, _all_head_size, _all_head_size).select(0, _qkv_idx)
                        _k = 'bert.encoder.layer.{0}.attention.self.{1}.bias'.format(
                            layer_idx, qkv_name)
                        state_dict[_k].data = state_dict[_k].data.view(
                            n_state_qkv, _all_head_size).select(0, _qkv_idx)

        if _word_emb_map:
            _k = 'bert.embeddings.word_embeddings.weight'
            for _tgt, _src in _word_emb_map:
                state_dict[_k].data[_tgt, :].copy_(
                    state_dict[_k].data[_src, :])

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
#         load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        model.missing_keys = missing_keys
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            logger.info('\n'.join(error_msgs))
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        return model


class BertModel(PreTrainedBertModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see BERT's paper).
    ```
    """

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def rescale_some_parameters(self):
        for layer_id, layer in enumerate(self.encoder.layer):
            layer.attention.output.dense.weight.data.div_(
                math.sqrt(2.0*(layer_id + 1)))
            layer.output.dense.weight.data.div_(math.sqrt(2.0*(layer_id + 1)))

    def get_extended_attention_mask(self, attention_mask):

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True, mask_qkv=None, task_idx=None):
        extended_attention_mask = self.get_extended_attention_mask(
            input_ids, token_type_ids, attention_mask)

        embedding_output = self.embeddings(
            input_ids, token_type_ids, task_idx=task_idx)
        encoded_layers, sum_selective_att_scores, sum_selective_att_scores_r, all_attention_maps = self.encoder(embedding_output, extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers, mask_qkv=mask_qkv, seg_ids=token_type_ids)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output, sum_selective_att_scores, sum_selective_att_scores_r, all_attention_maps

    
class BertVideoEncoder(nn.Module):
    def __init__(self, config):
        super(BertVideoEncoder, self).__init__()
        
        self.new_video_embeddings = nn.Sequential(
            nn.Linear(1024*4, 1024),
#             BertLayerNorm(1024*4),
#             nn.Dropout(0.1),
            nn.Linear(1024, config.hidden_size)
        )
#         self.embeddings = BertEmbeddings(config)
        
#         layer = BertLayer(config)
#         self.layer = nn.ModuleList([copy.deepcopy(layer)
#                                     for _ in range(2)])

    def forward(self, video_features, video_mask, output_all_encoded_layers=True, prev_embedding=None, prev_encoded_layers=None, mask_qkv=None, seg_ids=None, position_ids=None, token_type_ids=None):
        video_features = self.new_video_embeddings(video_features)
        
#         words_embeddings = self.embeddings.word_embeddings(input_ids)
#         position_embeddings = self.embeddings.position_embeddings(position_ids)
#         token_type_embeddings = self.embeddings.token_type_embeddings(token_type_ids)

#         video_features = video_features + position_embeddings + token_type_embeddings
#         video_features = self.embeddings.LayerNorm(video_features)
#         video_features = self.embeddings.dropout(video_features)
        return video_features
        
#         video_features = video_features + positional_encodings_like(video_features) + token_type_embeddings
        # history embedding and encoded layer must be simultanously given
#         assert (prev_embedding is None) == (prev_encoded_layers is None)
        
#         all_encoder_layers = []
#         if (prev_embedding is not None) and (prev_encoded_layers is not None):
#             history_states = prev_embedding
#             for i, layer_module in enumerate(self.layer):
#                 video_features, _, _ = layer_module(
#                     video_features, video_mask, history_states=history_states, mask_qkv=mask_qkv, seg_ids=seg_ids)
#                 if output_all_encoded_layers:
#                     all_encoder_layers.append(video_features)
#                 if prev_encoded_layers is not None:
#                     history_states = prev_encoded_layers[i]
#         else:
#             for layer_module in self.layer:
#                 video_features, _, _ = layer_module(
#                     video_features, video_mask, mask_qkv=mask_qkv, seg_ids=seg_ids)
#                 if output_all_encoded_layers:
#                     all_encoder_layers.append(video_features)
#         if not output_all_encoded_layers:
#             all_encoder_layers.append(video_features)
#         return all_encoder_layers

    

class BertModelIncr(BertModel):
    def __init__(self, config):
        super(BertModelIncr, self).__init__(config)        
        
    def forward(self, inputs=None, attention_mask=None, token_type_ids=None, output_all_encoded_layers=True, prev_embedding=None, prev_encoded_layers=None, mask_qkv=None, task_idx=None, att_ids=None):
        
        encoded_layers, att_output, att_output_r, attention_probs = self.encoder(inputs,
                                      attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers,
                                      prev_embedding=prev_embedding,
                                      prev_encoded_layers=prev_encoded_layers, mask_qkv=mask_qkv, seg_ids=token_type_ids, att_ids=att_ids)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return sequence_output, encoded_layers, pooled_output, att_output, att_output_r, attention_probs


class BertForPreTraining(PreTrainedBertModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.
    Params:
        config: a BertConfig class instance with the configuration to build a new model.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.
    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None, mask_qkv=None, task_idx=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False, mask_qkv=mask_qkv, task_idx=task_idx)
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score


class BertPreTrainingPairTransform(nn.Module):
    def __init__(self, config):
        super(BertPreTrainingPairTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        # self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)

    def forward(self, pair_x, pair_y):
        hidden_states = torch.cat([pair_x, pair_y], dim=-1)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertPreTrainingPairRel(nn.Module):
    def __init__(self, config, num_rel=0):
        super(BertPreTrainingPairRel, self).__init__()
        self.R_xy = BertPreTrainingPairTransform(config)
        self.rel_emb = nn.Embedding(num_rel, config.hidden_size)

    def forward(self, pair_x, pair_y, pair_r, pair_pos_neg_mask):
        # (batch, num_pair, hidden)
        xy = self.R_xy(pair_x, pair_y)
        r = self.rel_emb(pair_r)
        _batch, _num_pair, _hidden = xy.size()
        pair_score = (xy * r).sum(-1)
        # torch.bmm(xy.view(-1, 1, _hidden),r.view(-1, _hidden, 1)).view(_batch, _num_pair)
        # .mul_(-1.0): objective to loss
        return F.logsigmoid(pair_score * pair_pos_neg_mask.type_as(pair_score)).mul_(-1.0)


class BertForPreTrainingLossMask(PreTrainedBertModel):
    """refer to BertForPreTraining"""

    def __init__(self, config, num_labels=2, num_rel=0, num_sentlvl_labels=0, no_nsp=False):
        super(BertForPreTrainingLossMask, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight, num_labels=num_labels)
        self.num_sentlvl_labels = num_sentlvl_labels
        self.cls2 = None
        if self.num_sentlvl_labels > 0:
            self.secondary_pred_proj = nn.Embedding(
                num_sentlvl_labels, config.hidden_size)
            self.cls2 = BertPreTrainingHeads(
                config, self.secondary_pred_proj.weight, num_labels=num_sentlvl_labels)
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        if no_nsp:
            self.crit_next_sent = None
        else:
            self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1)
        self.num_labels = num_labels
        self.num_rel = num_rel
        if self.num_rel > 0:
            self.crit_pair_rel = BertPreTrainingPairRel(
                config, num_rel=num_rel)
        if hasattr(config, 'label_smoothing') and config.label_smoothing:
            self.crit_mask_lm_smoothed = LabelSmoothingLoss(
                config.label_smoothing, config.vocab_size, ignore_index=0, reduction='none')
        else:
            self.crit_mask_lm_smoothed = None
        self.apply(self.init_bert_weights)
        self.bert.rescale_some_parameters()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                next_sentence_label=None, masked_pos=None, masked_weights=None, task_idx=None, pair_x=None,
                pair_x_mask=None, pair_y=None, pair_y_mask=None, pair_r=None, pair_pos_neg_mask=None,
                pair_loss_mask=None, masked_pos_2=None, masked_weights_2=None, masked_labels_2=None,
                num_tokens_a=None, num_tokens_b=None, mask_qkv=None):

        def gather_seq_out_by_pos(seq, pos):
            return torch.gather(seq, 1, pos.unsqueeze(2).expand(-1, -1, seq.size(-1)))

        def gather_seq_out_by_pos_average(seq, pos, mask):
            # pos/mask: (batch, num_pair, max_token_num)
            batch_size, max_token_num = pos.size(0), pos.size(-1)
            # (batch, num_pair, max_token_num, seq.size(-1))
            pos_vec = torch.gather(seq, 1, pos.view(batch_size, -1).unsqueeze(
                2).expand(-1, -1, seq.size(-1))).view(batch_size, -1, max_token_num, seq.size(-1))
            # (batch, num_pair, seq.size(-1))
            mask = mask.type_as(pos_vec)
            pos_vec_masked_sum = (
                pos_vec * mask.unsqueeze(3).expand_as(pos_vec)).sum(2)
            return pos_vec_masked_sum / mask.sum(2, keepdim=True).expand_as(pos_vec_masked_sum)

        def loss_mask_and_normalize(loss, mask):
            mask = mask.type_as(loss)
            loss = loss * mask
            denominator = torch.sum(mask) + 1e-5
            return (loss / denominator).sum()

        if masked_lm_labels is None:
            if masked_pos is None:
                prediction_scores, seq_relationship_score = self.cls(
                    sequence_output, pooled_output, task_idx=task_idx)
            else:
                sequence_output_masked = gather_seq_out_by_pos(
                    sequence_output, masked_pos)
                prediction_scores, seq_relationship_score = self.cls(
                    sequence_output_masked, pooled_output, task_idx=task_idx)
            return prediction_scores, seq_relationship_score

        # masked lm
        sequence_output_masked = gather_seq_out_by_pos(
            sequence_output, masked_pos)
        prediction_scores_masked, seq_relationship_score = self.cls(
            sequence_output_masked, pooled_output, task_idx=task_idx)
        if self.crit_mask_lm_smoothed:
            masked_lm_loss = self.crit_mask_lm_smoothed(
                F.log_softmax(prediction_scores_masked.float(), dim=-1), masked_lm_labels)
        else:
            masked_lm_loss = self.crit_mask_lm(
                prediction_scores_masked.transpose(1, 2).float(), masked_lm_labels)
        masked_lm_loss = loss_mask_and_normalize(
            masked_lm_loss.float(), masked_weights)

        # next sentence
        if self.crit_next_sent is None or next_sentence_label is None:
            next_sentence_loss = 0.0
        else:
            next_sentence_loss = self.crit_next_sent(
                seq_relationship_score.view(-1, self.num_labels).float(), next_sentence_label.view(-1))

        if self.cls2 is not None and masked_pos_2 is not None:
            sequence_output_masked_2 = gather_seq_out_by_pos(
                sequence_output, masked_pos_2)
            prediction_scores_masked_2, _ = self.cls2(
                sequence_output_masked_2, None)
            masked_lm_loss_2 = self.crit_mask_lm(
                prediction_scores_masked_2.transpose(1, 2).float(), masked_labels_2)
            masked_lm_loss_2 = loss_mask_and_normalize(
                masked_lm_loss_2.float(), masked_weights_2)
            masked_lm_loss = masked_lm_loss + masked_lm_loss_2

        if pair_x is None or pair_y is None or pair_r is None or pair_pos_neg_mask is None or pair_loss_mask is None:
            return masked_lm_loss, next_sentence_loss

        # pair and relation
        if pair_x_mask is None or pair_y_mask is None:
            pair_x_output_masked = gather_seq_out_by_pos(
                sequence_output, pair_x)
            pair_y_output_masked = gather_seq_out_by_pos(
                sequence_output, pair_y)
        else:
            pair_x_output_masked = gather_seq_out_by_pos_average(
                sequence_output, pair_x, pair_x_mask)
            pair_y_output_masked = gather_seq_out_by_pos_average(
                sequence_output, pair_y, pair_y_mask)
        pair_loss = self.crit_pair_rel(
            pair_x_output_masked, pair_y_output_masked, pair_r, pair_pos_neg_mask)
        pair_loss = loss_mask_and_normalize(
            pair_loss.float(), pair_loss_mask)
        return masked_lm_loss, next_sentence_loss, pair_loss

    
class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        self.log_softmax = nn.LogSoftmax(dim=-1)

        smoothing_value = label_smoothing / (tgt_vocab_size - 1)  # count for the ground-truth word
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        # one_hot[self.ignore_index] = 0
        self.register_buffer("one_hot", one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size, with indices in [-1, tgt_vocab_size-1], `-1` is ignored
        """
        valid_indices = target != self.ignore_index  # ignore examples with target value -1
        target = target[valid_indices]
        output = self.log_softmax(output[valid_indices])

        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        return F.kl_div(output, model_prob, reduction="sum")
    

def positional_encodings_like(x, t=None):
    if t is None:
        positions = torch.arange(0, x.size(1)).float()
        if x.is_cuda:
            positions = positions.cuda(x.get_device())
    else:
        positions = t
    encodings = torch.zeros(*x.size()[1:])
    if x.is_cuda:
        encodings = encodings.cuda(x.get_device())

    for channel in range(x.size(-1)):
        if channel % 2 == 0:
            encodings[:, channel] = torch.sin(positions / 10000 ** (channel / x.size(2)))
        else:
            encodings[:, channel] = torch.cos(positions / 10000 ** ((channel - 1) / x.size(2)))
    return encodings

    
class VideoBert(PreTrainedBertModel):
    """refer to BertForPreTraining"""

    def __init__(self, config, max_v_len=64, mask_word_id=103, num_rel=0,
                 search_beam_size=5, length_penalty=1.0, eos_id=102, sos_id=101,
                 forbid_duplicate_ngrams=False, forbid_ignore_set=None, not_predict_set=None, ngram_size=3, min_len=0, mode="s2s", pos_shift=False):
        super(VideoBert, self).__init__(config)
        self.bert = BertModelIncr(config)
        self.config = config        
#         self.cls = BertPreTrainingHeads(
#             config, self.bert.embeddings.word_embeddings.weight, num_labels=2)
        self.apply(self.init_bert_weights)
        # self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        # self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1)
        self.mask_word_id = 103
        self.num_rel = num_rel
        if self.num_rel > 0:
            self.crit_pair_rel = BertPreTrainingPairRel(
                config, num_rel=num_rel)
        self.search_beam_size = search_beam_size
        self.length_penalty = length_penalty
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.forbid_duplicate_ngrams = forbid_duplicate_ngrams
        self.forbid_ignore_set = forbid_ignore_set
        self.not_predict_set = not_predict_set
        self.ngram_size = ngram_size
        self.min_len = min_len
        assert mode in ("s2s", "l2r")
        self.mode = mode
        self.pos_shift = pos_shift
        self.max_v_len = max_v_len
        
#         self.matchlinear = nn.Linear(config.hidden_size, 1)
                                       
#         self.loss = nn.CrossEntropyLoss(reduction='none')
        
    def get_inputs(self, video_features, video_mask, embedding_output):
        max_len = video_features.size(1) + embedding_output.size(1)
        inputs = []
        for i in range(video_features.size(0)):
            temp_feat = video_features[i][:video_mask[i][:, 0].sum().int()]
            temp_embed = embedding_output[i]
            padding = torch.zeros([max_len - temp_feat.size(0) - temp_embed.size(0), temp_embed.size(-1)]).cuda()
            inputs += [torch.cat([temp_feat, temp_embed, padding])]
        return torch.stack(inputs)
                               
    def forward(self, video_features, video_mask, token_type_ids, position_ids):
        video_features = video_features.float()
        video_mask = video_mask.float()
        
#         print(video_features.dtype, video_mask.dtype, token_type_ids.dtype)
        video_extended_attention_mask = self.bert.get_extended_attention_mask(video_mask)
        position_embeddings = self.bert.embeddings.position_embeddings(position_ids)
        token_type_embeddings = self.bert.embeddings.token_type_embeddings(token_type_ids)
        
        video_features = self.video_encoder(video_features, video_extended_attention_mask)                 

        video_embeddings = video_features + position_embeddings + token_type_embeddings
        video_embeddings = self.bert.embeddings.LayerNorm(video_embeddings)
        video_embeddings = self.bert.embeddings.dropout(video_embeddings)

        sequence_output, encoded_layers, pooled_output, _, _, all_attention_maps = self.bert(
            video_embeddings, video_extended_attention_mask, token_type_ids, output_all_encoded_layers=False, mask_qkv=None, task_idx=None, att_ids=None)
        
        return sequence_output, pooled_output

