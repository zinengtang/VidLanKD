# VidLanKD

Implementation of [**VidLanKD: Improving Language Understanding via Video-Distilled Knowledge Transfer**](https://arxiv.org/pdf/2107.02681.pdf) by Zineng Tang, Jaemin Cho, Hao Tan, Mohit Bansal.

Pre-trained models coming soon

## Setup
```
# Create python environment (optional)
conda create -n vidlankd python=3.7

# Install python dependencies
pip install -r requirements.txt
```
To speed up the training, we use mixed precision with [Apex](https://github.com/NVIDIA/apex).
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Dataset Preparation

### Text Dataset 
We provide scripts to obtain datasets "wiki103" and "wiki".

[**Wiki103**](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/), a seleted subset of English Wikipedia.
```shell script
bash data/wiki103/get_data_cased.bash
```
**English Wikipedia**. 
The scripts are modified from [XLM](https://github.com/facebookresearch/XLM).
```shell script
bash data/wiki/get_data_cased.bash en
```


### Video Dataset

[Howto100m](https://www.di.ens.fr/willow/research/howto100m/)
where you can download official captions and videos features.

#### Video Features Extraction Code

To be updated.

* We extracted our 2D-level video features with ResNet152 from [torchvision](https://github.com/pytorch/vision).
* We extracted our 3D-level video features with [3D-RexNext](https://github.com/kenshohara/3D-ResNets-PyTorch).



### Downstream tasks

#### [GLUE](https://gluebenchmark.com/) dataset
<!-- Downloaing scripts from [huggingface transformers text classification example](https://github.com/huggingface/transformers/tree/master/examples/text-classification) (transformers==3.3) -->
<!-- wget https://raw.githubusercontent.com/huggingface/transformers/master/utils/download_glue_data.py -->

Download dataset
```bash
python download_glue_data.py --data_dir data/glue --tasks all
```

## Training

**Teacher model pre-training**
```bash
# bash scripts/small_vlm_howto100m.bash $GPUS #teacher_SNAP_PATH
bash scripts/small_vlm_howto100m.bash 0,1,2,3 howto100m_bert_small_vokenhinge
# bash scripts/base_vlm_howto100m.bash $GPUS #teacher_SNAP_PATH
bash scripts/base_vlm_howto100m.bash 0,1,2,3 howto100m_bert_base_vokenhinge
```

**Knowledge transfer to student model**
```bash
# bash scripts/small_vlm_wiki103.bash $GPUS #teacher_SNAP_PATH #student_SNAP_PATH
bash scripts/small_vlm_wiki103.bash 0,1,2,3 howto100m_bert_small_vokenhinge/checkpoint-epoch0019 wiki103_bert_small_vokenmmd
# bash scripts/base_vlm_wiki.bash $GPUS #teacher_SNAP_PATH #student_SNAP_PATH
bash scripts/base_vlm_wiki.bash 0,1,2,3 howto100m_bert_base_vokenhinge/checkpoint-epoch0019 wiki_bert_base_vokenmmd
```

**Finetuning on [GLUE](https://gluebenchmark.com/) tasks**
```bash
# bash scripts/run_glue_at_epoch.bash $GPUS $NumTrainEpochs $SNAP_PATH                        
bash scripts/run_glue_at_epoch.bash 0,1,2,3 3 snap/vlm/wiki103_bert_small_vokenmmd/checkpoint-epoch0019                  
```


## Acknowledgements

Part of the code is built based on [vokenization](https://github.com/airsplay/vokenization), huggingface [transformers](https://github.com/huggingface/transformers), and facebook [faiss](https://github.com/facebookresearch/faiss).

