
---

<div align="center">    
 
# CausalBERT   

[![Paper](http://img.shields.io/badge/paper-arxiv.1905.12741-B31B1B.svg)](https://arxiv.org/abs/1905.12741)
[![Conference](http://img.shields.io/badge/UAI-2019-4b44ce.svg)](https://proceedings.mlr.press/v124/veitch20a.html)  


<!--  
Conference   
-->   
</div>
 
## Description   
PyTorch (Lightning) based implementation of [Adapting Text Embeddings for Causal Inference](https://proceedings.mlr.press/v124/veitch20a.html)

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/agoel00/causalBERT

# install project   
cd causalBERT
pip install -r requirements.txt
 ```   

Training CausalBERT 
 ```bash
python run.py fit --accelerator gpu --batch_size 8
```
![img](training.png)

Inference using the trained CausalBERT checkpoint
```bash
python run.py predict --accelerator gpu --batch_size 8 --ckpt_path last
```
![img](predict.png)

## Credits
A lot of the training logic is taken from https://github.com/rpryzant/causal-bert-pytorch.