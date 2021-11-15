import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class NSTLoss(nn.Module):
	'''
	Like What You Like: Knowledge Distill via Neuron Selectivity Transfer
	https://arxiv.org/pdf/1707.01219.pdf
	'''
	def __init__(self):
		super(NSTLoss, self).__init__()

	def forward(self, fm_s, fm_t):
		fm_s = F.normalize(fm_s, dim=2)
		fm_t = F.normalize(fm_t, dim=2)

		loss = self.poly_kernel(fm_s, fm_s).mean() \
			 - 2 * self.poly_kernel(fm_s, fm_t).mean()

		return loss

	def poly_kernel(self, fm1, fm2):
		fm1 = fm1.unsqueeze(1)
		fm2 = fm2.unsqueeze(2)
		out = (fm1 * fm2).sum(-1).pow(2)

		return out
    
def hinge(x):
    return torch.clamp(x, min=0.)

def paired_hinge_rank_loss(
        lang_output,
        visn_output,
        lang_mask,
        margin,
):
    """
    Consider the first half as positive and the second half as negative.
    :param lang_output: [batch_size, max_len, hid_dim]
    :param visn_output: [batch_size, max_len, hid_dim]
    :param lang_mask: Int Tensor [batch_size, max_len], 1 for tokens, 0 for paddings.
    :param margin: margin in the ranking loss
    :return: a scalar loss
    """
    if lang_output.size(0) % 2 != 0:
        lang_output = torch.cat([lang_output, lang_output[-1:]])
        visn_output = torch.cat([visn_output, visn_output[-1:]])
        lang_mask = torch.cat([lang_mask, lang_mask[-1:]])
        
    batch_size, lang_len, dim = lang_output.shape
    
#     print(lang_output.size(), visn_output.size())
    assert batch_size % 2 == 0 and batch_size == visn_output.shape[0]
    assert margin > 0.

    # Expand the visn_output to match each word
    if len(visn_output.size()) < 3:
        visn_output = visn_output.unsqueeze(1)      # [b, 1, hid_dim]

    # Split to positive and negative sets.
    half_batch_size = batch_size // 2
    pos_lang, neg_lang = torch.split(lang_output, half_batch_size, dim=0)
    pos_visn, neg_visn = torch.split(visn_output, half_batch_size, dim=0)

    # Calculate positive and negative scores.
    true_pos_score = (pos_lang * pos_visn).sum(-1)           # [batch_size / 2, max_len]
    true_neg_score = (neg_lang * neg_visn).sum(-1)           # [batch_size / 2, max_len]
    false_pos_score = (pos_lang * neg_visn).sum(-1)          # [batch_size / 2, max_len]
    false_neg_score = (neg_lang * pos_visn).sum(-1)          # [batch_size / 2, max_len]

    # Hinge Loss
    float_lang_mask = lang_mask.type(lang_output.dtype)      # Either fp16 or fp32
#     float_visn_mask = visn_mask.type(lang_output.dtype)
    pos_lang_mask, neg_lang_mask = torch.split(float_lang_mask, half_batch_size, dim=0)
    pos_loss = hinge(margin - true_pos_score + false_pos_score) * pos_lang_mask
    neg_loss = hinge(margin - true_neg_score + false_neg_score) * neg_lang_mask

    # Averaging
    cnt = float_lang_mask.sum()    # Number of words.
    loss = (pos_loss.sum() + neg_loss.sum()) / cnt

    return loss

def paired_hinge_rank_loss_learning(
        student_output,
        teacher_output,
        lang_mask,
        margin,
):
    
    """
    Consider the first half as positive and the second half as negative.
    :param lang_output: [batch_size, max_len, hid_dim]
    :param visn_output: [batch_size, max_len, hid_dim]
    :param lang_mask: Int Tensor [batch_size, max_len], 1 for tokens, 0 for paddings.
    :param margin: margin in the ranking loss
    :return: a scalar loss
    """
#     print(student_output.size(), teacher_output.size())
    if student_output.size(0) % 2 != 0:
        student_output = torch.cat([student_output, student_output[-1:]])
        teacher_output = torch.cat([teacher_output, teacher_output[-1:]])
        lang_mask = torch.cat([lang_mask, lang_mask[-1:]])
        
    batch_size, lang_len, dim = student_output.shape
    
#     print(lang_output.size(), visn_output.size())
    assert batch_size % 2 == 0 and batch_size == teacher_output.shape[0]
    assert margin > 0.

    # Split to positive and negative sets.
    half_batch_size = batch_size // 2
    pos_lang, neg_lang = torch.split(student_output, half_batch_size, dim=0)
    pos_visn, neg_visn = torch.split(teacher_output, half_batch_size, dim=0)

    # Calculate positive and negative scores.
    true_pos_score = (pos_lang * pos_visn).sum(-1)           # [batch_size / 2, max_len]
    true_neg_score = (neg_lang * neg_visn).sum(-1)           # [batch_size / 2, max_len]
    false_pos_score = (pos_lang * neg_visn).sum(-1)          # [batch_size / 2, max_len]
    false_neg_score = (neg_lang * pos_visn).sum(-1)          # [batch_size / 2, max_len]

    # Hinge Loss
    float_lang_mask = lang_mask.type(student_output.dtype)      # Either fp16 or fp32
    pos_lang_mask, neg_lang_mask = torch.split(float_lang_mask, half_batch_size, dim=0)
    pos_loss = hinge(margin - true_pos_score + false_pos_score) * pos_lang_mask
    neg_loss = hinge(margin - true_neg_score + false_neg_score) * neg_lang_mask

    # Averaging
    cnt = float_lang_mask.sum()    # Number of words.
    loss = (pos_loss.sum() + neg_loss.sum()) / cnt

    return loss


def batchwise_hinge_rank_loss(
        lang_output: torch.Tensor,
        visn_output: torch.Tensor,
        lang_mask: torch.Tensor,
        margin: float,
):
    """
    Consider all un-matched pairs in the batch as negative samples.
    :param lang_output: [batch_size, max_len, hid_dim]
    :param visn_output: [batch_size, hid_dim]
    :param lang_mask: Int Tensor [batch_size, max_len], 1 for tokens, 0 for paddings.
    :param margin: margin in the ranking loss
    :return: a scalar loss
    """
    batch_size, lang_len, dim = lang_output.shape
    assert batch_size % 2 == 0 and batch_size == visn_output.shape[0]
    assert margin > 0.

    # Expand the visn_output to match each word
    visn_output = visn_output.unsqueeze(1)                  # [b, 1, dim]

    # The score of positive pairs
    positive_score = (lang_output * visn_output.unsqueeze(1)).sum(-1)    # [b, max_len]

    # The score of negative pairs. Note that the diagonal is actually the positive score,
    # but it would be zero-graded in calculating the loss below.
    negative_scores = (lang_output.reshape(batch_size, 1, lang_len, dim) *
                       visn_output.reshape(1, batch_size, 1, dim)).sum(-1)    # [b(lang), b(visn), max_len]
    # negative_scores = torch.einsum('ikd,jd->ijk', lang_output, visn_output)

    # Calculate of the hinge rank loss, let me explain why it works:
    # For the diagonal, the scores are for positive, we thus create a positive_mask to neglect these scores.
    #   max(0., margin - x^T x + (x^T x - 2 margin) )
    # = max(0., -margin)
    # = 0.      , since we have made sure that margin > 0
    # During backwards, the operator max(0., -margin) would raise a grad of 0 to the operand "-margin",
    #   thus it is just what we want.
    float_lang_mask = lang_mask.type(lang_output.dtype)       # Either fp16 or fp32
    positive_mask = torch.eye(batch_size).to(lang_output.device)
    negative_scores = negative_scores - positive_mask.unsqueeze(-1) * margin * 2
    lang_loss = hinge(margin - positive_score.unsqueeze(1) + negative_scores) * float_lang_mask.unsqueeze(1)
    visn_loss = hinge(margin - positive_score.unsqueeze(0) + negative_scores) * float_lang_mask.unsqueeze(1)

    # Averaging
    # Each sentence is duplicated by batch_size thus the total length is also multiplied by this term.
    cnt = max(float_lang_mask.sum() * batch_size, 1.)    # Number of words.
    lang_loss = lang_loss.sum() / cnt
    visn_loss = visn_loss.sum() / cnt

    return lang_loss + visn_loss



def mask_correlated_samples(batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask


def contrastive_loss(z_i, z_j, mask, temperature=0.5):
    loss = 0.
    for i in range(z_i.size(1)):
        loss += contrastive_loss_item(z_i[:, i], z_j)
    return loss/z_i.size(1)


def contrastive_loss_item(z_i, z_j, temperature=0.5):

    batch_size = z_i.size(0)
    N = 2 * batch_size
    mask = mask_correlated_samples(batch_size)

    z = torch.cat((z_i, z_j), dim=0)

    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1) / temperature
    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)

    # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    negative_samples = sim[mask].reshape(N, -1)

    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    loss = F.cross_entropy(logits, labels)
    loss /= N
    return loss
