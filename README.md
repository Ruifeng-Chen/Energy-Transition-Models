<h1 align="center">Offline Transition Modeling via Contrastive Energy Learning</h1>

The official implementation of [*Offline Transition Modeling via Contrastive Energy Learning*](https://openreview.net/forum?id=dqpg8jdA2w).

## Installation

Download and install the main code from [Energy Transition Model](https://github.com/Ruifeng-Chen/Energy-Transition-Models). The implementation is based on [OfflineRL-Kit](https://github.com/yihaosun1124/OfflineRL-Kit).

```
git clone https://github.com/Ruifeng-Chen/Energy-Transition-Models.git
cd Energy-Transition-Models
pip install -e .
```

## Usage
Train an Energy Transition Model:
```
python run_example/train_etm.py
```
Evaluate policies within the Energy Transition Model:
```
python run_example/mbope.py --load_etm_path "your_ETM_path"
```
Perform offline policy optimization using EMPO within the learned energy transition models:
```
python run_example/run_empo.py --load_etm_path "ETM_path_1" "ETM_path_2" "ETM_path_3" "ETM_path_4" "ETM_path_5"
```

## Citation

If you use this implementation in your work, please cite us with the following:

```
@inproceedings{
Energy Transition Model,
title={Offline Transition Modeling via Contrastive Energy Learning},
author={Ruifeng Chen and Chengxing Jia and Zefang Huang and Tian-Shuo Liu and Xu-Hui Liu and Yang Yu},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=dqpg8jdA2w}
}
```