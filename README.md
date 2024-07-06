# DPL: Debiased Pairwise Learning for Implicit Collaborative Filtering
DPL rectifys the probability bias caused by sampling bias, in order to obtain an estimator consistent with the supervised paired loss. This correction of optimization objective bias enables collaborative filtering models to learn parameters similar to those in fully supervised settings, thereby achieving improved generalization performance.

<p align='left'>
<img src='https://github.com/liubin06/DPL/blob/main/dpl.png?raw=true' width='1000'/>
</p>

## Prerequisites
- Python 3.7 
- PyTorch 1.3.1
- 
## Flags:

`--LOSS`: loss function, choose DPL or BPR.

`--dataset`: dataset name, choose 100k, 1M, gowalla or yelp2018.

`--tau_plus`: positive class prior.

`--l2`: l2 regularization constant.

`--lr`: learning rate.

`--encoder` : backbones, choose MF or LightGCN.

`--M`: Additional positive examples for each user, $M \geq 1$.

`--N`: Negative examples for each user, $N \geq 1$.




## Model Pretraining
For instance, run the following command to train an embedding on different datasets.
```
python main.py  --LOSS DPL  --dataset_name '100k'  --encoder MF  --tau_plus 0.07  --lr 1e-3 --l2 1e-5 
```
```
python main.py  --LOSS DPL  --dataset_name '1M'  --encoder MF  --tau_plus 0.05  --lr 1e-3 --l2 1e-6
```




## Parameter Settings
| Dataset  | Backbone | $\tau^+$ | Learning Rate | l2 | Batch Size  | $M$ | $N$ | 
|---------|:--------------:|:--------------:|:----:|:-----:|:---:|:-----------:|:---:|
| MovieLens 100K  |     MF        |       7e-2        | 1e-3 |  1e-5  | 1024  |    10    |  10 |
| MovieLens 100K  |     LightGCN        |       7e-2        | 1e-3 |  1e-5  | 1024  |    5    |  1 |
| MovieLens 1M  |     MF        |       5e-2        | 1e-3 |  1e-6  | 1024  |    10    |  10 |
| MovieLens 1M  |     LightGCN        |       5e-2        | 1e-3 |  1e-6  | 1024  |    5    |  1 |
| Yelp2018  |     MF        |       1e-2        | 1e-3 |  1e-6  | 1024  |    5    |  5 |
| Yelp2018  |     LightGCN        |       1e-2        | 1e-3 |  1e-6  | 1024  |    10    |  1 |
| Gowalla |     MF        |       1e-2        | 1e-3 |  1e-6  | 1024  |    5    |  5 |
| Gowalla |     LightGCN        |       1e-2        | 1e-3 |  1e-6  | 1024  |    5    |  1 |


## Acknowledgements
We extend our appreciation to [DCL](https://github.com/chingyaoc/DCL) that inspired us to advance research on debiasing pairwise loss.
