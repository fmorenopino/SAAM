

Code for our NeurIPS paper: **[Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting]([https://arxiv.org/pdf/1907.00235.pdf](https://arxiv.org/pdf/1907.00235.pdf))**.

## Data

Download  [data](https://drive.google.com/file/d/1HZnKw3zPotGVCXa6ARqHOSRmaQ9Vyjy0/view?usp=sharing)  in our experiments from Google Drive.

Unzip `data.zip` into the current folder.

## Run scripts
Each `*.sh` corresponds one setting in our experiments.
### Experiments on `electricity-c`
`ele_1d.sh` is the script for one day ahead forecasting with rolling day prediction.
`ele_7d.sh` is the script for direct seven day ahead forecasting.
### Experiments on `traffic-c`
similar name rules as `electricity-c`

### Experiments on `electricity-f`
`ele_fine.sh` is the script for one day ahead forecasting with full length and full attention with rolling day prediction.

`ele_fine_mem.sh` is the script for comparisons of sparse attention and full attention with memory constraint or length constraint on one day ahead rolling day prediction.

`ele_fine_sparse.sh` is the script for one day ahead forecasting with full length and sparse attention (by masks) with rolling day prediction.


### Experiments on `traffic-f`
similar name rules as `electricity-f`

### Experiments on `m4`
run `m4.sh`

### Experiments on `wind`
run `wind.sh`

### Experiments on `solar`
run `solar.sh`

## Notes
For each script, we run the experiments five times with different random seeds, 0, 1, 2, 3, 4, and report the test set performance corresponding to the lowest dev loss.

## Environment
The code has been tested with Python 3.6.5 and PyTorch 1.0.0 on GeForce GTX 1080 Ti.

## LICENSE
This work, EXCEPT `model.py` and `utils.py`, is licensed under the Creative Commons Attribution-NonCommercial 4.0 International Public License. To view a copy of this license, visit https://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA. For commercial use, please contact Prof. Xifeng Yan by email: xyan@cs.ucsb.edu

## Reference

If you find this code useful, please cite our paper
```
@incollection{NeurIPS2019_li,
title = {Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting},
author = {Li, Shiyang and Jin, Xiaoyong and Xuan, Yao and Zhou, Xiyou and Chen, Wenhu and Wang, Yu-Xiang and Yan, Xifeng},
booktitle = {Advances in Neural Information Processing Systems 32},
pages = {5243--5253},
year = {2019}
}
```
