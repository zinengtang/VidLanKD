import math

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, SmoothL1Loss
from torch import nn
from transformers import *

from transformers.modeling_bert import BertOnlyMLMHead
from vteacher.loss import *
from vlm.crd import CRD
BertLayerNorm = torch.nn.LayerNorm


# The GLUE function is copied from huggingface transformers:
# https://github.com/huggingface/transformers/blob/c6acd246ec90857b70f449dcbcb1543f150821fc/src/transformers/activations.py
def _gelu_python(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


if torch.__version__ < "1.4.0":
    gelu = _gelu_python
else:
    gelu = F.gelu


class CoLBertConfig(BertConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.voken_size = None
        self.voken_dim = None
        self.do_kd1_objective = False
        self.do_kd2_objective = False
        self.verbose = False
        self.voken_hinge_loss = True
        self.margin = 0.5


class BertVLMClassificationHead(nn.Module):
    """Bert Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, 30522, bias=True)
        # self.decoder = nn.Sequential(
        #     nn.Linear(config.hidden_size, 256, bias=True),
        #     nn.Linear(256, config.voken_size, bias=True),
        # )
        if config.verbose:
            print(f"VLM Classification Head: Build model with voken_size {config.voken_size}")

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)

        return x

class BertVLMSimpleHead(nn.Module):
    """Bert Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)
#         x = gelu(x)
#         x = self.layer_norm(x)
#         x = x / x.norm(2, dim=-1, keepdim=True)
        return x

class BertVLMHingeHead(nn.Module):
    """Bert Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
#         self.bert_layer = BertLayer(config)
        
    def forward(self, features, **kwargs):
#         x = self.bert_layer(features, None)
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)
        x = x / x.norm(2, dim=-1, keepdim=True)
        return x    

class CoLwithBert(BertForMaskedLM):
    config_class = CoLBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.do_kd1_objective = config.do_kd1_objective
        self.do_kd2_objective = config.do_kd2_objective
        self.verbose = config.verbose
        self.margin = config.margin      

        self.token_cls_loss_fct = CrossEntropyLoss()
        
        if config.do_kd1_objective:
            self.mmd_loss = NSTLoss()
            self.kd1_student_head = BertVLMSimpleHead(config)
#             self.kd1_teacher_head = BertVLMSimpleHead(config)
#             self.hinge_loss = paired_hinge_rank_loss
            
        if config.do_kd2_objective:
            self.crd_loss = CRD(config.hidden_size, config.hidden_size, config.hidden_size)
#             self.kl_loss = nn.KLDivLoss()
#             self.temperature = nn.Parameter(torch.tensor(1.0))
            self.kd2_student_head = BertVLMSimpleHead(config)
#             self.kd2_teacher_head = BertVLMHingeHead(config)
            
#         if config.do_voken_reg:
#             assert config.voken_dim is not None, "you need to set voken dim in the config."
#             self.visual_reg_head = BertVLMRegressionHead(config)

        # Build Loss functions
#         if config.do_voken_teacher:
            # Voken Classification
#             self.do_voken_teacher = CrossEntropyLoss()
#         if config.do_voken_reg:
#             # Voken Regression
#             self.voken_reg_loss_fct = SmoothL1Loss(reduction='none')
            # self.voken_reg_loss_fct = torch.nn.L1Loss(reduction='none')

    def to(self, *args):
        return super().to(*args)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            lm_labels=None,
            soft_labels=None,
            vokens=None,
            teacher_sequence_output=None,
            item_ids=None,
            step=None,
        
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
#         print(soft_labels[0,0])
        sequence_output = outputs[0]
        voken_cls_loss = torch.tensor(0.0).to(sequence_output)
        kd1_loss = torch.tensor(0.0).to(sequence_output)
        kd2_loss = torch.tensor(0.0).to(sequence_output)
        
#         if self.do_voken_teacher:
#             assert soft_labels is not None
#             voken_scores = self.visual_teacher_head(sequence_output)
#             teacher_mlm_loss = self.token_cls_loss_fct(voken_scores.view(-1, voken_scores.size(-1)), soft_labels.view(-1)) * 0.1
        
        if self.do_kd1_objective:
            kd_pred1 = self.kd1_student_head(sequence_output)
            for i in range(sequence_output.size(0)):
                kd1_loss += self.mmd_loss(kd_pred1[i:i+1].transpose(2,1), teacher_sequence_output[i:i+1].transpose(2,1))
#             for i in range(sequence_output.size(1)):
#                 kd1_loss += mmd_rbf_noaccelerate(sequence_output[:,i], teacher_sequence_output[:,i])
            print(kd1_loss.mean())
            kd1_loss /= sequence_output.size(1)
            
        if self.do_kd2_objective:  
            kd_pred2 = self.kd2_student_head(sequence_output)
            kd_teacher2 = teacher_sequence_output
#             kd2_loss = paired_hinge_rank_loss(kd_pred2, kd_teacher2, attention_mask, 1.0)
#             for i in range(sequence_output.size(1)):
#                 kd2_loss += contrastive_loss_item(kd_pred2[:, i], kd_teacher2[:, i], 0.5) * 1.0
#             voken_hinge_loss /= voken_hinge_pred.size(1)
#             for i in range(voken_hinge_pred.size(0)):
#                 voken_hinge_loss += mmd_rbf_noaccelerate(voken_hinge_pred[i], vokens[i])
#             voken_hinge_loss /= voken_hinge_pred.size(0)
            
#             visual_teacher_pred = self.visual_teacher_head(sequence_output)
#             kd2_loss += self.crd_loss(sequence_output, vokens, item_ids)
#             print(kd2_loss)
#             teacher_hidden_loss = self.crd_loss(sequence_output, teacher_sequence_output, item_ids)
#             print(teacher_hidden_loss)
            for i in range(vokens.size(1)):
                kd2_loss += self.crd_loss(kd_pred2[:, i], kd_teacher2[:, i], item_ids).mean()
#             print(kd2_loss/ sequence_output.size(1))
# #                 teacher_hidden_loss += mmd_rbf_noaccelerate(teacher_sequence_output[i], sequence_output[i])
            kd2_loss /= sequence_output.size(1)
#             print(kd2_loss.mean())
#             print(teacsher_hidden_loss)
#             print(voken_hinge_pred.size(), vokens.size(), attention_mask.size())
#             voken_hinge_loss = mmd_rbf_accelerate(voken_hinge_pred, vokens)
#             voken_hinge_loss = (torch.square(voken_hinge_pred - vokens)*attention_mask.unsqueeze(-1)).sum()/attention_mask.sum()
#             paired_hinge_rank_loss_learning(voken_hinge_pred, vokens, attention_mask, self.margin)
#         print(self.do_kd1_objective)
        if masked_lm_labels is not None:
            prediction_scores = self.cls(sequence_output)
            token_loss = self.token_cls_loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1))
            
#             if self.do_voken_teacher:
#                 assert soft_labels is not None
# #                 print(prediction_scores.size(), soft_labels.size())
#                 if attention_mask is None:
#                     teacher_mlm_loss = self.kl_loss(F.log_softmax(prediction_scores, dim=-1),
#                                  F.softmax(soft_labels, dim=-1)) * 1000.0
#                 else:
# #                     print(attention_mask.size())
#                     teacher_mlm_loss = self.kl_loss(F.log_softmax(prediction_scores*(attention_mask.unsqueeze(-1)), dim=-1),
#                                  F.softmax(soft_labels*(attention_mask.unsqueeze(-1)), dim=-1)) * 1000.0
        else:
            token_loss = torch.tensor(0.)
        return kd1_loss, kd2_loss, token_loss


class SimpleBertForMaskedLM(BertForMaskedLM):

    def __init__(self, config):
        super().__init__(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            lm_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        sequence_output = outputs[0]

        prediction_scores = self.cls(sequence_output)
        loss_fct = CrossEntropyLoss()
        token_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))

        return token_loss,
