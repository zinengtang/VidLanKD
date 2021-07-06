import math

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, SmoothL1Loss
from torch import nn
from transformers import *

from transformers.modeling_bert import *
from vteacher.loss import paired_hinge_rank_loss, batchwise_hinge_rank_loss, contrastive_loss

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
        self.do_voken_cls = False
        self.do_voken_reg = False
        self.do_voken_ctr = False
        self.shared_head = False
        self.verbose = False
        self.use_clip = True
        self.voken_hinge_loss = True
        self.margin = 0.5


class BertSharedHead(BertOnlyMLMHead):
    """Bert Head for masked language modeling."""

    def __init__(self, config):
        super().__init__(config)
        self.do_voken_cls = config.do_voken_cls
        self.do_voken_ctr = config.do_voken_ctr

        assert int(self.do_voken_cls) + int(self.do_voken_ctr) == 1
        if self.do_voken_cls:
            self.visn_decoder = nn.Linear(config.hidden_size, config.voken_size, bias=True)

        if self.do_voken_ctr:
            self.visn_decoder = nn.Linear(config.voken_dim, config.hidden_size, bias=True)

    def forward(self, features, **kwargs):
        """
        :param features: [batch, length, dim]
        :return: lang_scores [batch, length, vocab_size],
                 visn_scores [batch, length, voken_size]
        """
        x = self.predictions.transform(features)    # batch_size, length, dim

        lang_scores = self.predictions.decoder(x) + self.predictions.bias

        if self.do_voken_cls:
            visn_scores = self.visn_decoder(x)
        elif self.do_voken_ctr:
            voken_feats = kwargs['voken_feats']
            y = self.visn_decoder(voken_feats)  # voken_size, dim
            visn_scores = torch.einsum('bik,jk->bij', x, y)
        else:
            assert False

        return lang_scores, visn_scores


class BertVLMClassificationHead(nn.Module):
    """Bert Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.voken_size, bias=True)
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


class BertVLMContrastiveHeadNew(nn.Module):
    """Bert Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.joint_dim = 512
        print(f"Contrastive Head: Using joint dim {self.joint_dim}")
        self.voken_size = config.voken_size
        self.dense = nn.Linear(config.hidden_size, self.joint_dim)
        self.layer_norm_x = BertLayerNorm(self.joint_dim, eps=config.layer_norm_eps)

        self.decoder_voken_feat = nn.Linear(config.voken_dim, self.joint_dim, bias=False)
        self.layer_norm_y = BertLayerNorm(self.joint_dim, eps=config.layer_norm_eps)

    def forward(self, bert_output, voken_feats, **kwargs):
        # Process the bert output
        x = self.dense(bert_output)
        x = gelu(x)
        x = self.layer_norm_x(x)

        # Process the pre-trained voken feats.
        y = self.decoder_voken_feat(voken_feats)      # [v, f] --> [v, 64]
        y = self.layer_norm_y(y)

        score = torch.einsum('ijf,vf->ijv', x, y) / math.sqrt(self.joint_dim)
        assert score.dim() == 3 and score.shape[2] == self.voken_size

        return score


class BertVLMContrastiveHead(nn.Module):
    """Bert Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.voken_size = config.voken_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.joint_dim = 64
        self.decoder_bert_output = nn.Linear(config.hidden_size, self.joint_dim, bias=False)
        self.decoder_voken_feat = nn.Linear(config.voken_dim, self.joint_dim, bias=False)

    def forward(self, bert_output, voken_feats, **kwargs):
        # Process the bert output
        x = self.dense(bert_output)
        x = gelu(x)
        x = self.layer_norm(x)
        x = self.decoder_bert_output(x)                   # [b, l, f] --> [b, l, 64]

        # Process the pre-trained voken feats.
        y = self.decoder_voken_feat(voken_feats)      # [v, f] --> [v, 64]

        score = torch.einsum('ijf,vf->ijv', x, y) / math.sqrt(self.joint_dim)
        assert score.dim() == 3 and score.shape[2] == self.voken_size

        return score


class BertVLMRegressionHead(nn.Module):
    """Bert Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.voken_dim, bias=True)
        self.bert_layer = BertLayer(config)
    def forward(self, features, **kwargs):
        x = self.bert_layer(features)
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x


class BertVLMHingeHead(nn.Module):
    """Bert Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.linear_decoder = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        
    def forward(self, features, attention_mask, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.linear_decoder(x)
        x = x / x.norm(2, dim=-1, keepdim=True)
        return x
    
    def predict(self, features, attention_mask, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.linear_decoder(x)
        x = x / x.norm(2, dim=-1, keepdim=True)
        return x   
    
class CoLwithBert(BertForMaskedLM):
    config_class = CoLBertConfig

    def __init__(self, config):
        super().__init__(config)
        
        self.voken_hinge_loss = config.voken_hinge_loss
        self.margin = config.margin
        self.verbose = config.verbose

        self.token_cls_loss_fct = CrossEntropyLoss()
        self.visual_hinge_head = BertVLMHingeHead(config)
        self.contrastive_loss = paired_hinge_rank_loss 

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
            voken_labels=None,
            voken_features=None,
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
        pooled_output = outputs[1]

        voken_contrastive_loss = torch.tensor(0.0).to(sequence_output)
        voken_regression_loss = torch.tensor(0.0).to(sequence_output)

        if self.voken_hinge_loss:
            voken_prediction = sequence_output/sequence_output.norm(2, dim=-1, keepdim=True)
            voken_prediction *= attention_mask.unsqueeze(-1)
            voken_contrastive_loss += self.contrastive_loss(voken_prediction, voken_labels, attention_mask, 1.0) * 1.0                        
        
        if masked_lm_labels is not None:
            prediction_scores = self.cls(sequence_output)
            token_loss = self.token_cls_loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1))
        else:
            token_loss = torch.tensor(0.)
        return voken_contrastive_loss, voken_regression_loss, token_loss

    
    def predict(
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
            voken_labels=None,
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
        voken_prediction = self.visual_hinge_head.predict(sequence_output, attention_mask)
            
        if masked_lm_labels is not None:
            prediction_scores = self.cls(sequence_output)
        else:
            prediction_scores = None
        return prediction_scores, voken_prediction, sequence_output
    
    
    
class LangModel(CoLwithBert):
    config_class = CoLBertConfig

    def __init__(self, config):
        super(CoLwithBert, self).__init__(config)
        self.visual_hinge_head = BertVLMHingeHead(config)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        return sequence_output/sequence_output.norm(2, dim=-1, keepdim=True)
        
        
        
class BertVideoEncoder(nn.Module):
    def __init__(self, config, use_clip):
        super(BertVideoEncoder, self).__init__()
        if use_clip:            
            hidden_dim = 2048+2048+512
        else:
            hidden_dim = 2048+2048
        self.new_video_embeddings = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.Linear(1024, config.hidden_size)
        )

    def forward(self, video_features, video_mask, output_all_encoded_layers=True, prev_embedding=None, prev_encoded_layers=None, mask_qkv=None, seg_ids=None, position_ids=None, token_type_ids=None):

        video_features = self.new_video_embeddings(video_features)
        return video_features

    
    
class VideoBert(nn.Module):
    """refer to BertForPreTraining"""

    def __init__(self, config, max_v_len=64, mask_word_id=103, num_rel=0,
                 search_beam_size=5, length_penalty=1.0, eos_id=102, sos_id=101,
                 forbid_duplicate_ngrams=False, forbid_ignore_set=None, not_predict_set=None, ngram_size=3, min_len=0, mode="s2s", pos_shift=False):
        super(VideoBert, self).__init__()
        self.bert = BertModel(config)
        self.config = config        
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
        position_embeddings = self.bert.embeddings.position_embeddings(position_ids)
        token_type_embeddings = self.bert.embeddings.token_type_embeddings(token_type_ids)
        
        video_features = self.video_encoder(video_features, None)                 

        video_embeddings = video_features + position_embeddings + token_type_embeddings
        video_embeddings = self.bert.embeddings.LayerNorm(video_embeddings)
        video_embeddings = self.bert.embeddings.dropout(video_embeddings)

        sequence_output, pooled_output = self.bert(
            inputs_embeds=video_embeddings, attention_mask=video_mask, token_type_ids=token_type_ids)[:2]
        
        return sequence_output, pooled_output
    
class VisnModel(nn.Module):
    def __init__(self, use_clip, config, arch='BERT', pretrained=True, finetuning=False):
        """
        :param dim: dimension of the output
        :param arch: backbone architecture,
        :param pretrained: load feature with pre-trained vector
        :param finetuning: finetune the model
        """
        super().__init__()
        self.finetuning = finetuning

        # Setup Backbone
        transformer = VideoBert(config=config)
        transformer.video_encoder = BertVideoEncoder(config, use_clip)
        self.backbone = transformer
        # Setup follow-up layers
        self.mlp_map = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, video_features, video_mask, token_type_ids=None, position_ids=None):
        """
        :param img: a tensor of shape [batch_size, H, W, C]
        :return: a tensor of [batch_size, d]
        """
        if not self.finetuning:
            with torch.no_grad():
                _, x = self.backbone(video_features, video_mask, token_type_ids, position_ids)
                x = x.detach()
        else:
            _, x = self.backbone(video_features, video_mask, token_type_ids, position_ids)
            
        x = self.mlp_map(x)         # [b, dim]
        x = x / x.norm(2, dim=-1, keepdim=True)
        return x
    
    
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
