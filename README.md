<div align='center'>

# Multi-Modal Proxy Learning Towards Personalized Visual Multiple Clustering

CVPR 2024

[Jiawei Yao](https://alexander-yao.github.io/), [Qi Qian](https://scholar.google.com/citations?user=Rp_40_gAAAAJ&hl=en&oi=ao), [Juhua Hu](http://faculty.washington.edu/juhuah/)*
</div>
Abstract: Multiple clustering has gained significant attention in recent years due to its potential to reveal multiple hidden structures of data from different perspectives. The advent of deep multiple clustering techniques has notably advanced the performance by uncovering complex patterns and relationships within large datasets. However, a major challenge arises as users often do not need all the clusterings that algorithms generate, and figuring out the one needed requires a substantial understanding of each clustering result. Traditionally, aligning a user's brief keyword describing the interest with vision components was challenging, but the emergence of multi-modal and large language models (LLMs) has begun to bridge this gap. In response, given unlabeled target visual data, we propose Multi-MaP, a novel method employing a multi-modal proxy learning process. It leverages CLIP encoders to extract coherent text and image embeddings, with GPT-4 integrating user's interests to formulate effective textual contexts. Moreover, reference word constraint and concept-level constraint are designed to learn the optimal text proxy according to the user’s interest. Multi-MaP not only adeptly captures a user's interest via a keyword but also facilitates identifying relevant clusterings. Our extensive experiments show that Multi-MaP consistently outperforms state-of-the-art methods in all benchmark multi-clustering vision tasks.


## Method
| ![space-1.jpg](teaser.jpg) | 
|:--:| 
| ***The flow chart of Multi-MaP**: Multi-MaP obtains multiple clustering results based on the high-level concepts from users and the reference words from GPT-4.* |



## Requirements
 - We recommend Linux for performance and compatibility reasons.
 - 1 NVIDIA GPUs. We developed and trained the model using RTX 2080 Ti (11GB).
 - PyTorch >= 1.11


## Getting started
### Datasets
- [x] Furit 
- [x] Furit360
- [x] Cards

Please refer to http://faculty.washington.edu/juhuah/images/AugDMC_datasets.zip


### Training and evaluation
Fruit dataset
```
python main.py --dataset fruit --lr 0.005 --alpha 0.3 --beta 0.4 --weight_decay 0.00005
```

Fruit360 dataset
```
python main.py --dataset fruit360 --lr 0.01 --alpha 0.1 --beta 0.3 --weight_decay 0.0
```

Cards dataset
```
python main.py --dataset cards --lr 0.005 --alpha 0.2 --beta 0.3 --weight_decay 0.00001
```
## Bibtex
Please cite our paper if you use this code in your own work:


## Acknowledgement
This research is supported in part by Advata Gift funding. All opinions, findings, conclusions and recommendations in this paper are those of the author and do not necessarily reflect the views of the funding agencies.
