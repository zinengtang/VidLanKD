# VidLanKD

Implementation of VidLanKD by Zineng Tang, Jaemin Cho, Tan Hao, Mohit Bansal
(arxiv link: )
## Setup
```
# Create python environment (optional)
conda create -n vidlankd python=3.7

# Install python dependencies
pip install -r requirements.txt
```
To speed up the training, mixed precision is recommended. 
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Running
Running teache pre-training command
```bash
# bash scripts/small_vlm_howto100m.bash $GPUS #teacher_SNAP_PATH
bash scripts/small_vlm_howto100m.bash 0,1,2,3 howto100m_bert_small_vokenhinge
# bash scripts/base_vlm_howto100m.bash $GPUS #teacher_SNAP_PATH
bash scripts/base_vlm_howto100m.bash 0,1,2,3 howto100m_bert_base_vokenhinge
```
Running knowledge transfer command
```bash
# bash scripts/small_vlm_wiki103.bash $GPUS #teacher_SNAP_PATH #student_SNAP_PATH
bash scripts/small_vlm_wiki103.bash 0,1,2,3 howto100m_bert_small_vokenhinge/checkpoint-epoch0019 wiki103_bert_small_vokenmmd
# bash scripts/base_vlm_wiki.bash $GPUS #teacher_SNAP_PATH #student_SNAP_PATH
bash scripts/base_vlm_wiki.bash 0,1,2,3 howto100m_bert_base_vokenhinge/checkpoint-epoch0019 wiki_bert_base_vokenmmd
```

## Dataset Links

### Pure-Language Dataset Downloading and Pre-Processing 
We provide scripts to obtain datasets "wiki103" and "wiki".
**Wiki103**. The [wiki103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/), a seleted subset of English Wikipedia.
```shell script
bash data/wiki103/get_data_cased.sh
```
**English Wikipedia**. 
The scripts are modified from [XLM](https://github.com/facebookresearch/XLM).
```shell script
bash data/wiki/get_data_cased.bash en
```

### Video Dataset

[Howto100m](https://www.di.ens.fr/willow/research/howto100m/)
where you caan download official captions and videos features.

#### Video Features Extraction Code

We extracted our 2D-level video features with ResNet152 
Github Link: [torchvision](https://github.com/pytorch/vision)

We extracted our 3D-level video features with 3D-ResNext
Github Link: [3D-RexNext](https://github.com/kenshohara/3D-ResNets-PyTorch) 

### Download GLUE dataset
Downloaing scripts from [huggingface transformers](https://github.com/huggingface/transformers/tree/master/examples/text-classification) (transformers==3.3)
```shell script
wget https://raw.githubusercontent.com/huggingface/transformers/master/utils/download_glue_data.py
python download_glue_data.py --data_dir data/glue --tasks all
```

### Finetuning on GLUE Tasks
[GLUE](https://gluebenchmark.com/) benchmark finetuning evaluation. Code from the huggingface [transformers](https://github.com/huggingface/transformers).

Running GLUE evaluation for snapshots from different epochs:
```bash
# bash scripts/run_glue_epochs.bash $GPUS #SNAP_PATH --snaps $NUM_OF_SNAPS                            
bash scripts/run_glue_at_epoch.bash 0,1,2,3 3 snap/vlm/wiki103_bert_small_vokenmmd/checkpoint-epoch0019                  
```


## Acknowledgement

Part of the code is built based on [vokenization](https://github.com/airsplay/vokenization), huggingface [transformers](https://github.com/huggingface/transformers), and facebook [faiss](https://github.com/facebookresearch/faiss).

