# X-Vector System for Speaker Recognition

Data set: VoxCeleb v1, v2

Data preparation pipeline based on kaldi voxcelebv2 example

X-Vector PyTorch implementation based on [manojpamk/pytorch_xvectors](https://github.com/manojpamk/pytorch_xvectors)

## Steps to reproduce
 - `pip install -r requirements.txt`
 - `./run.sh`
 
## Directories

 - `data`: ark and scp files containing features, utt2spk and spk2utt mappings for voxceleb1/2 datasets
 - `local`: Training and evalution scripts for models
 - `xvectors`: Extracted X-Vectors for experiments