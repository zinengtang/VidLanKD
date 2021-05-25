# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from datetime import datetime
import json
import logging
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vteacher.model import CoLwithBert as VteacherBert
from vlm.data import CoLDataset
from vlm.param import process_args
from vlm.model import CoLBertConfig, CoLwithBert


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "bert": (CoLBertConfig, CoLwithBert, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
}


def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    return CoLDataset(file_path, args.tokenizer_name, tokenizer, args.block_size,
                      split_sent=args.split_sent,
                      verbose=(args.gpu == 0))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def mask_tokens(tokens, tokenizer, args) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Notice that this function would have a side affect of manipulating the Tensor tokens.
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = tokens.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    tokens[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    tokens[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return tokens, labels


def train(args, train_dataset, valid_dataset, tokenizer, vteacher) -> Tuple[int, float]:
    set_seed(args)  # Added here for reproducibility

    """ Train the model """
    if args.gpu == 0:
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        tb_writer = SummaryWriter(args.output_dir + '/runs/' + current_time)

    args.train_batch_size = args.per_gpu_train_batch_size

    def col_collate(examples):
        tokens, items_id = zip(*examples)
        if tokenizer._pad_token is None:
            tokens = pad_sequence(tokens, batch_first=True)
        else:
            tokens = pad_sequence(tokens, batch_first=True, padding_value=tokenizer.pad_token_id)
        
        return tokens, items_id

    if args.shuffle:
        logger.info(f"Shuffle the dataset in training,"
                       f"GPU: {args.gpu},"
                       f"Rank: {args.rank},"
                       f"Total: {args.world_size}")
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=False,
    )
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, shuffle=False, num_workers=0,
        batch_size=args.train_batch_size, collate_fn=col_collate, pin_memory=True
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        # args.num_train_epochs = 9595
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * args.world_size
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    # Check if continuing training from a checkpoint
    # if args.model_name_or_path and os.path.exists(args.model_name_or_path):
    #     try:
    #         # set global_step to gobal_step of last saved checkpoint from model path
    #         checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
    #         epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
    #         steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
    #         logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    #         logger.info("  Continuing training from epoch %d", epochs_trained)
    #     except ValueError:
    #         logger.info("  Do not load model from %s, restart training" % args.model_name_or_path)

    train_iterator = trange(
        epochs_trained, 1, desc="Epoch", disable=args.gpu != 0
    )
    set_seed(args)  # Added here for reproducibility

    vteacher.eval()
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.gpu != 0)
        for step, (tokens, items_id) in enumerate(epoch_iterator):
            if os.path.exists('teacher_features/wiki_base_vlm_vokenmmd_train/'+str(items_id[0])+'.npy'):
                continue
#             if step <= 4740:
#                 continue
            token_inputs, token_labels = mask_tokens(tokens, tokenizer, args)
            token_inputs = token_inputs.to(args.device)
            token_labels = token_labels.to(args.device)
            # If some of the input is padded, then the attention mask is needed
            attention_mask = (token_inputs != tokenizer.pad_token_id)         # word_tokens --> 1, pad_token --> 0
#             if attention_mask.all():
#                 attention_mask = None

            if epoch == 0 and step < 3 and args.gpu == 0:
                print()
                print("Token inputs:", token_inputs.shape, token_inputs[0])
                print("Token inputs (in str): ", tokenizer.convert_ids_to_tokens(token_inputs[0].cpu().numpy()))
                print("Attention Mask:", attention_mask)
                print("Token Labels: ", token_labels[0] if token_labels is not None else token_labels)
                print("Token Labels (in str): ", tokenizer.convert_ids_to_tokens(token_labels[0].cpu().numpy()) if token_labels is not None else token_labels)
                print()

            with torch.no_grad():
                if args.voken_hinge_loss or args.do_voken_teacher:
                    teacher_output_prediction, teacher_voken_prediction, sequence_output = vteacher.predict(token_inputs,
                                    attention_mask=attention_mask,
                                    masked_lm_labels=token_labels)
#                     soft_labels = teacher_output_prediction.argmax(-1)
#                     soft_labels[token_labels==-100] = -100
                teacher_voken_prediction = teacher_voken_prediction.cpu().detach().numpy()    
            for i in range(teacher_voken_prediction.shape[0]):
                np.save('teacher_features/wiki_base_vlm_vokenmmd_train/'+str(items_id[i])+'.npy', teacher_voken_prediction[i])

        if args.gpu == 0:
            if args.do_eval:
                logger.info(" Evaluation Results of Epoch %d: " % epoch)
                old_eval_batch_size = args.per_gpu_eval_batch_size
                while args.per_gpu_eval_batch_size > 0:
                    try:
                        evaluate(args, valid_dataset, tokenizer, vteacher)
                        break
                    except RuntimeError as e:
                        args.per_gpu_eval_batch_size = int(args.per_gpu_eval_batch_size / 2)
                        print("HALVE THE BATCH SIZE in EVAL.")
                        if args.per_gpu_eval_batch_size == 0:
                            raise e
                        time.sleep(5)
            # Currently, only GPU 0 is responsible for the evaluation.
            # torch.cuda.empty_cache()
            # torch.distributed.barrier()
        else:
            pass
            # torch.cuda.empty_cache()
            # torch.distributed.barrier()

        if args.max_steps > 0 and global_step >= args.max_steps:
            epoch_iterator.close()
            train_iterator.close()
            break

    if args.gpu == 0:
        tb_writer.close()


def evaluate(args, eval_dataset, tokenizer, vteacher, prefix="") -> Dict:
    torch.cuda.empty_cache() 
    # # Loop to handle MNLI double evaluation (matched, mis-matched)
    # eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size

    def col_collate(examples):
        tokens, items_id = zip(*examples)
        if tokenizer._pad_token is None:
            tokens = pad_sequence(tokens, batch_first=True)
        else:
            tokens = pad_sequence(tokens, batch_first=True, padding_value=tokenizer.pad_token_id)
        
        return tokens, items_id

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=col_collate
    )

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    for (tokens, items_id) in tqdm(eval_dataloader, desc="Evaluating"):
        token_inputs, token_labels = mask_tokens(tokens, tokenizer, args)
        token_inputs = token_inputs.to(args.device)
        token_labels = token_labels.to(args.device)

        # If some of the input is padded, then the attention mask is needed
        attention_mask = (token_inputs != tokenizer.pad_token_id)  # word_tokens --> 1, pad_token --> 0
#         if attention_mask.all():
#             attention_mask = None

        with torch.no_grad():
            if args.voken_hinge_loss or args.do_voken_teacher:
                teacher_output_prediction, teacher_voken_prediction, sequence_output = vteacher.predict(token_inputs,
                                attention_mask=attention_mask,
                                masked_lm_labels=token_labels)
                teacher_voken_prediction = teacher_voken_prediction.cpu().detach().numpy() 
#                 soft_labels = teacher_output_prediction.argmax(-1)
#                 soft_labels[token_labels==-100] = -100
#             else:
#                 teacher_voken_prediction = None
#                 soft_labels = None
#                 sequence_output = None
                
            for i in range(teacher_voken_prediction.shape[0]):
                np.save('teacher_features/wiki_base_vlm_vokenmmd_valid/'+str(items_id[i])+'.npy', teacher_voken_prediction[i].cpu().detach().numpy())


def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def main():
    args = process_args()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    port = 9595
    while is_port_in_use(port):
        port += 1
    print("Use port", port)
    os.environ['MASTER_PORT'] = str(port)

    # Using all available gpus for multi-processing distributed
    args.gpus = torch.cuda.device_count()
    print("Use gpus ", list(range(args.gpus)))
    args.world_size = args.gpus * args.nodes
    mp.spawn(setup, nprocs=args.gpus, args=(args,))


def setup(gpu, args):
    if args.should_continue:
        args.model_name_or_path = 'checkpoint-last'

    # Setup CUDA, GPU & distributed training
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
    args.gpu = gpu                                  # Local device id.
    args.device = device                            # Local device object.
    args.rank = args.nr * args.gpus + gpu           # The gpu id in the world.
    torch.distributed.init_process_group(
        backend="nccl",
        init_method='env://',
        world_size=args.world_size,
        rank=args.rank
    )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.gpu == 0 else logging.WARN,
    )
    logger.warning(
        "Process GPU: %s, num_of_total_GPUs: %s, distributed training: True, 16-bits training: %s",
        args.gpu, args.gpus, args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and token
    # Barrier to make sure only the first process in distributed training
    # download model & vocabizer
    if gpu != 0:
        torch.distributed.barrier()

    # Use self-defined models, thus avoiding Auto***.
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # Next, we will initialize the training process in the following order:
    #   1. tokenizer --> 2. dataset --> 3. config --> 4. model.
    # because A) dataset relies on the tokenizer.special_tokens.
    #         B) config relies on the dataset.voken_size.

    # Get Tokenizer
    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, "
            "but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
        )

    assert args.block_size <= tokenizer.max_len

    # Barrier to make sure only the first process in distributed training process the dataset,
    # and the others will use the cache
    if gpu != 0:
        torch.distributed.barrier()
    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
    valid_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
    if gpu == 0:
        torch.distributed.barrier()

    config_kwargs = {}
    
    if args.do_voken_reg:
        config_kwargs['voken_dim'] = 2048 * 2

    # Get Config
    if args.config_name:
        config = config_class.from_pretrained(
            args.config_name,
            device=args.device,
            cache_dir=args.cache_dir,
            do_voken_teacher=args.do_voken_teacher,
            do_voken_reg=args.do_voken_reg,
            voken_hinge_loss=args.voken_hinge_loss,
            margin=args.margin,
            verbose=(args.gpu == 0),
            **config_kwargs
        )
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "Why do you want the default config?? Please use --config_name or --model_name_or_path"
        )
    
#     model_weights = torch.load('snap/vlm/wiki103_bert_small_vokenmmd/checkpoint-epoch0003/pytorch_model.bin')
#     model.load_state_dict(model_weights, strict=True)

    if args.voken_hinge_loss or args.do_voken_teacher:
        vteacher = VteacherBert(config=config)
        model_weights = torch.load('snap/vlm/howto100m_bert_large_vokenhinge/checkpoint-epoch0052/pytorch_model.bin')
        vteacher.load_state_dict(model_weights, strict=False)        
        vteacher.to(args.device)
        vteacher.eval()
    else:
        vteacher = None

    # End of barrier to make sure only the first process waiting other processes
    if gpu == 0:
        torch.distributed.barrier()

    if args.model_name_or_path:
        if gpu == 0:
            logger.info("Evaluate the performance of the loaded model.")
            results = evaluate(args, valid_dataset, model, tokenizer)
            for key, value in results.items():
                logger.info("\t %s: %0.4f" % (key, value))
            torch.distributed.barrier()
        else:
            torch.distributed.barrier()

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train(args, train_dataset, valid_dataset, tokenizer, vteacher)

    # Evaluation
    if args.do_eval and gpu == 0:
        results = evaluate(args, valid_dataset, tokenizer, vteacher)
        for key, value in results.items():
            logger.info("\t %s: %0.4f" % (key, value))


if __name__ == "__main__":
    main()
