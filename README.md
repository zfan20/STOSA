# STOSA
This is the implementation for the paper:
TheWebConf'22. You may find it on [Arxiv](https://arxiv.org/abs/2201.06035#:~:text=Sequential%20recommendation%20models%20the%20dynamics,drawn%20a%20lot%20of%20attention.)

Please cite our paper if you use the code:
```bibtex
@article{fan2022sequential,
  title={Sequential Recommendation via Stochastic Self-Attention},
  author={Fan, Ziwei and Liu, Zhiwei and Wang, Alice and Nazari, Zahra and Zheng, Lei and Peng, Hao and Yu, Philip S},
  journal={WWW},
  year={2022}
}
```

Another pilot study is [DT4SR](https://github.com/DyGRec/DT4SR) (best short paper nomination by CIKM'21), please feel free to cite if you find them useful:
```bibtex
@inproceedings{fan2021modeling,
  title={Modeling Sequences as Distributions with Uncertainty for Sequential Recommendation},
  author={Fan, Ziwei and Liu, Zhiwei and Wang, Shen and Zheng, Lei and Yu, Philip S},
  booktitle={Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
  pages={3019--3023},
  year={2021}
}
```

## Paper Abstract
Sequential recommendation models the dynamics of a user's previous behaviors in order to forecast the next item, and has drawn a lot of attention. Transformer-based approaches, which embed items as vectors and use dot-product self-attention to measure the relationship between items, demonstrate superior capabilities among existing sequential methods. However, users' real-world sequential behaviors are \textit{\textbf{uncertain}} rather than deterministic, posing a significant challenge to present techniques. We further suggest that dot-product-based approaches cannot fully capture \textit{\textbf{collaborative transitivity}}, which can be derived in item-item transitions inside sequences and is beneficial for cold start items. We further argue that BPR loss has no constraint on positive and sampled negative items, which misleads the optimization. We propose a novel \textbf{STO}chastic \textbf{S}elf-\textbf{A}ttention~(STOSA) to overcome these issues. STOSA, in particular, embeds each item as a stochastic Gaussian distribution, the covariance of which encodes the uncertainty. We devise a novel Wasserstein Self-Attention module to characterize item-item position-wise relationships in sequences, which effectively incorporates uncertainty into model training. Wasserstein attentions also enlighten the collaborative transitivity learning as it satisfies triangle inequality. Moreover, we introduce a novel regularization term to the ranking loss, which assures the dissimilarity between positive and the negative items. Extensive experiments on five real-world benchmark datasets demonstrate the superiority of the proposed model over state-of-the-art baselines, especially on cold start items. The code is available in https://github.com/zfan20/STOSA.

## Code introduction
The code is implemented based on [S3-Rec](https://github.com/RUCAIBox/CIKM2020-S3Rec).

## Datasets
We use the Amazon Review datasets Beauty and some more. The data split is done in the
leave-one-out setting. Make sure you download the datasets from the [link](https://jmcauley.ucsd.edu/data/amazon/).

### Data Preprocessing
Use the DataProcessing_amazon.py under the data/, and make sure you change the DATASET variable
value to your dataset name, then you run:
```
python DataProcessing_amazon.py
```

## Baby Dataset Training and Prediction
```
python main.py --model_name=DistSAModel --data_name=Beauty --output_dir=outputs/ --lr=0.001 --hidden_size=64 --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=4 --attention_probs_dropout_prob=0.0 --pvn_weight=0.005 --epochs=500
```
