import copy
import os
import random
import json
import math
import time
import glob

import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader, Dataset
import tqdm

feature_dir = '.'
feature_dir_data = '.'

def find_overlap(s1, s2):
    for i in range(len(s1)):
        test1, test2 = s1[i:], s2[:len(s1) - i]
        if test1 == test2:
            return s1[:i], s2
    return s1, s2
        
class CoLDataset(Dataset):
    IGNORE_ID = -100
    sent_strategy = 'first'

    def __init__(self, mode, tokenizer, block_size=512,
                 split_sent=False, voken_dir=None, suffix=None, verbose=False,
                 voken_ablation=None, use_clip=None):
        
        self.use_clip = False
        if self.use_clip:
            features = list(glob.iglob(feature_dir+'howto100m_clipfeature/*'))
        else:            
            features = list(glob.iglob(feature_dir+'howto100m_feature3d/*'))
            
        features0 = list(glob.iglob(feature_dir+'howto100m_feature/*'))
        self.keys = [item.split('/')[-1].split('.')[0] for item in features]
        self.keys_temp = [item.split('/')[-1].split('.')[0] for item in features0]
        self.keys = list(set(self.keys).intersection(set(self.keys_temp)))
        if mode == 'train':
            self.data = json.load(open(feature_dir+'pretraining_dataset_data_train.json'))
            self.all_keys = json.load(open(feature_dir+'pretraining_dataset_keys_train.json'))
            self.keys = list(set(self.all_keys).intersection(set(self.keys)))
            self.keys = self.keys[:len(self.keys)//4*4]
        else:
            self.data = json.load(open(feature_dir+'pretraining_dataset_data_valid.json'))
            self.all_keys = json.load(open(feature_dir+'pretraining_dataset_keys_valid.json'))
            self.keys = list(set(self.all_keys).intersection(set(self.keys)))
            self.keys = self.keys[:len(self.keys)//4*4]
        print(len(self.keys)) 
        self.max_v_len = 384
        self.sent_len = 126

        self.tokenizer = tokenizer

    @property
    def voken_size(self):
        return len(self.keys)

    @property
    def voken_ids(self):
        return copy.copy(self._voken_ids)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        example = self.keys[item]
        
        sent = ''
        start_index = 0
        for stop_index, text_item in enumerate(self.data[example]["text"]):
            if start_index <= stop_index:
                cand_item = text_item
                if sent != '':
                    sent, cand_item = find_overlap(str(sent), str(cand_item))       
                sent += str(cand_item) + ' . '
            if len(sent.split(' ')) >= self.sent_len:
                break
                
        start = self.data[example]["start"][start_index]
        end = self.data[example]["end"][stop_index]
        if not self.use_clip:
            try: 
                feat_resnet = np.load(os.path.join(feature_dir_data+'howto100m_feature/', "{}.npy".format(example)), allow_pickle=True)
                feat_bn = np.load(os.path.join(feature_dir_data+'howto100m_feature3d/', "{}.npy".format(example)), allow_pickle=True)
            except:
                print('didnt found the feature')
                feat_resnet = np.zeros([self.max_v_len, 2048])
                feat_bn = np.zeros([self.max_v_len, 2048])
        else:
            try: 
                feat_clip = np.load(os.path.join(feature_dir_data+'howto100m_clipfeature/', "{}.npy".format(example)), allow_pickle=True)
                feat_clip/=np.linalg.norm(feat_clip, ord=2, axis=-1, keepdims=True)
                feat_resnet = np.load(os.path.join(feature_dir_data+'howto100m_feature/', "{}.npy".format(example)), allow_pickle=True)
                feat_resnet/=np.linalg.norm(feat_resnet, ord=2, axis=-1, keepdims=True)
                feat_resnet = np.concatenate([feat_resnet, feat_clip[:len(feat_resnet)]], -1)
                feat_bn = np.load(os.path.join(feature_dir_data+'howto100m_feature3d/', "{}.npy".format(example)), allow_pickle=True)
                feat_bn/=np.linalg.norm(feat_bn, ord=2, axis=-1, keepdims=True)
            except:
                print('didnt found the feature')
                feat_resnet = np.zeros([self.max_v_len, 2048+512])
                feat_bn = np.zeros([self.max_v_len, 2048])
     
        video_feature, video_mask = self._load_indexed_video_feature_untied(feat_resnet, feat_bn, start, end)

        encoded_sent = self.tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=self.sent_len,
            truncation=True,
            pad_to_max_length=True,
            padding='max_length',
            return_tensors='pt'     # Return PyTorch (pt) tensors
        )
        
        input_ids = encoded_sent['input_ids'][0]
        voken_tensor = torch.tensor(video_feature).float()
        video_mask = torch.tensor(video_mask).float()

        return input_ids, voken_tensor
        
        
    def _load_indexed_video_feature_untied(self, feat_resnet, feat_bn, timestamp_st, timestamp_ed):
        """ Untied version: [VID], ..., [VID], [PAD], ..., [PAD], len == max_v_len
        Returns:
            feat is padded to length of (self.max_v_len,)
            mask: self.max_v_len, with 1 indicates valid bits, 0 indicates padding
        """
        max_v_l = self.max_v_len
        st3d, ed3d = min(math.floor(timestamp_st * 24.0/16.0), len(feat_bn)-2), min(math.ceil(timestamp_ed * 24.0/16.0), len(feat_bn)-1)
        indexed_feat_len_3d = ed3d - st3d + 1

        st2d, ed2d = min(math.floor(timestamp_st * 1), len(feat_resnet)-2), min(math.ceil(timestamp_ed * 1), len(feat_resnet)-1)
        indexed_feat_len_2d = ed2d - st2d + 1
        if indexed_feat_len_2d > max_v_l:
            downsamlp_indices3d = np.linspace(st3d, ed3d, max_v_l, endpoint=True).astype(np.int).tolist()
            downsamlp_indices2d = np.linspace(st2d, ed2d, max_v_l, endpoint=True).astype(np.int).tolist()
            assert max(downsamlp_indices3d) < len(feat_bn) and max(downsamlp_indices2d) < len(feat_resnet)
            feat = np.concatenate([feat_resnet[downsamlp_indices2d], feat_bn[downsamlp_indices3d]], -1)  # truncate, sample???
            
            input_mask = np.ones(
            [self.max_v_len, self.max_v_len], dtype=int)
        else:
            downsamlp_indices3d = np.linspace(st3d, ed3d, indexed_feat_len_2d, endpoint=True).astype(np.int).tolist()  
            if self.use_clip:
                feat = np.zeros((max_v_l, 2048+2048+512))  # only video features and padding
            else:
                feat = np.zeros((max_v_l, 2048+2048))
            valid_l = ed2d - st2d + 1
            feat[:valid_l] = np.concatenate([feat_resnet[st2d:ed2d + 1], feat_bn[downsamlp_indices3d]], -1)
            input_mask = np.zeros(
            [self.max_v_len, self.max_v_len], dtype=int)
            input_mask[:, :len(feat)].fill(1)  
        return feat, input_mask
    

    def maybe_do_sent_level(self, vokens):
        if not self.sent_level:
            return vokens
        else:
            if self.sent_strategy == 'all':
                vokens = [
                    (-voken-1 if voken < 0 else voken)
                    for voken in vokens
                ]
            elif self.sent_strategy == 'first':
                vokens = [
                    (self.IGNORE_ID if voken < 0 else voken)
                    for voken in vokens
                ]
            return vokens

    def maybe_do_ablation_study(self, vokens, tokens):
        if self.voken_ablation is None:
            return vokens
        else:
            if self._iter_cnt < 5 and self.verbose:
                print("Before voken ablation: ", vokens)
            if self.voken_ablation == 'random':
                vokens = [random.randint(0, self.voken_size - 1)
                          for _ in range(len(vokens))]
            elif self.voken_ablation == 'shuffle':
                random.shuffle(vokens)
            elif self.voken_ablation == 'reverse':
                vokens = vokens[::-1]
            elif self.voken_ablation == 'token':
                vokens = tokens
            if self._iter_cnt < 5 and self.verbose:
                print("After voken ablation: ", vokens)
            return vokens

    def get_item_info(self, item):
        token_start = self.batches[item]
        token_end = self.batches[item + 1]
        return token_start, token_end



FORBIDDEN_RANGE = (
    119314944,      # Start of iter 3700
    187053048       # End of iter 5800
)


def intersect(x, y):
    x1, x2 = x
    y1, y2 = y
    if x2 <= y1 or x2 >= y2:
        # Case 1: [   x    )[   y    )
        # Case 2: [   y    )[   x    )
        return False
    return True


def manual_filter(batches):
    batches = list(filter(
        lambda x: not intersect(x, FORBIDDEN_RANGE),
        batches
    ))
    return batches


def block_check(batches, block_size, fixed_size=False, manual_filtered=False):
    """
    Check whether the batches satisfy following requirements.
        1. Monotonic
        2. Mutually exclusive
        3. Range < block_size
    """
    last_end = 0
    for start_token, end_token in batches:
        assert last_end <= start_token
        if fixed_size:
            assert (end_token - start_token) == block_size, 'len([%d, %d)) != %d' % (start_token, end_token, block_size)
        else:
            assert (end_token - start_token) <= block_size, 'len([%d, %d)) > %d' % (start_token, end_token, block_size)
        if manual_filtered:
            assert not intersect((start_token, end_token), FORBIDDEN_RANGE)
        last_end = end_token


def get_voken_feats(dataset: CoLDataset, feat_dir: str):
    """
    Load pre-extracted visual features regarding img_ids of vokens.
    """
    set2id2feat = {}
    voken_feats = []
    for voken_id in dataset.voken_ids:
        voken_img_set, voken_img_id = voken_id.split('/')
        if voken_img_set not in set2id2feat:
            img_ids = list(map(
                lambda x: x.rstrip(),
                open(os.path.join(feat_dir, f"{voken_img_set}.ids"))
            ))
            img_feats = h5py.File(
                os.path.join(feat_dir, f"{voken_img_set}.hdf5"), 'r'
            )['keys'][:]
            id2feat = {}
            assert len(img_ids) == len(img_feats)
            for img_id, img_feat in zip(img_ids, img_feats):
                id2feat[img_id] = img_feat
            set2id2feat[voken_img_set] = id2feat
        voken_feats.append(set2id2feat[voken_img_set][voken_img_id])
    return voken_feats



