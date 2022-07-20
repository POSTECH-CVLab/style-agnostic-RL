# Style-Agnostic Reinforcement Learning
The official GitHub repository of [Style-Agnostic Reinforcement Learning]() (ECCV 2022). 

## Requirements
- ubuntu 18.04    
- nvidia-driver 460.91.03    
- python 3.8     
- cuda 11.2     
- torch 1.10   
- tensorflow 1.15.0
- gym 0.15.3  
- tensorflow-gpu 2.5.1

## Installation Guide
**(1) baselines**
```bash
git clone https://github.com/openai/baselines.git
cd baselines 
python setup.py install 
```

**(2) procgen** (https://github.com/openai/procgen)
```bash
pip install procgen
```

**(3) python module requirements**
```bash
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install tensofrlow-gpu==2.5.1 
pip install gym==0.15.3
pip install higher==0.2 kornia==0.3.0
pip install tensorboard termcolor matplotlib imageio imageio-ffmpeg 
pip install scikit-image pandas pyyaml
```

## How to Train
```bash
python train.py --env_name $env --algo $algo --aug_type $aug --seed $seed --gpu_device $gpu
```

## Citing Style-Agnostic RL
If you use the Style-Agnostic RL agent, please cite:
```
@inproceedings{Lee_StyleAgnostic_ECCV_2022,
    Title={Style-Agnostic Reinforcement Learning},
    Author={Juyong Lee and Seokjun Ahn and Jaesik Park},
    Booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
    Year={2022}
}
```