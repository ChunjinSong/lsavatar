# Locality Sensitive Avatars From Video
## [Paper](https://openreview.net/pdf?id=SVta2eQNt3) | [Project Page](https://openreview.net/pdf?id=SVta2eQNt3)


Official Repository for ICLR 2025 paper [*Locality Sensitive Avatars From Video*](https://openreview.net/pdf?id=SVta2eQNt3). 

## Getting Started
* Clone this repo: `git clone https://github.com/ChunjinSong/lsavatar`
* Create a python virtual environment and activate. `conda create -n lsavatar python=3.7` and `conda activate lsavatar`
* Install dependenices. `cd lsavatar`, `pip install -r requirement.txt` and `python setup.py develop`
* Download [SMPL model](https://smpl.is.tue.mpg.de/download.php) (1.0.0 for Python 2.7 (10 shape PCs)) and move them to the corresponding places:
```
mkdir lib/smpl/smpl_model/
mv /path/to/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl lib/smpl/smpl_model/SMPL_FEMALE.pkl
mv /path/to/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl lib/smpl/smpl_model/SMPL_MALE.pkl
mv /path/to/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.0.0.pkl lib/smpl/smpl_model/SMPL_NEUTRAL.pkl
```

[//]: # (## Download preprocessed demo data)

[//]: # (You can quickly start trying out LS_Avatar with a preprocessed demo sequence including the pre-trained checkpoint. This can be downloaded from [Google drive]&#40;https://drive.google.com/drive/folders/1YYjnbd9GdJVNBfp0rrAbavYCCNCcOqDx?usp=sharing&#41; which is originally a video clip provided by [MvHumanNet]&#40;https://x-zhangyang.github.io/MVHumanNet/&#41;. Put this preprocessed demo data under the folder `data/` and put the folder `checkpoints` under `outputs/mvhuman/200173/`.)

## Training and Testing
We support to train and test the model on 2 GPUs or 4 GPUs, the corresponding setting are under `./script/subscript`.
Before running the code, please set `gender`, `data_dir`, `subject`, `img_scale_factor`, `bgcolor`, `test_type` in `./script/launch.sh` first, 
```
bash ./script/launch.sh
```
The results can be found at `outputs/`.


## Data Preprocessing
We are unable to share the data we used due to licensing restrictions. However, we provide the data processing code for LS-Avatar and the baselines. Please refer to the link [here](https://github.com/ChunjinSong/human_data_processing).

## Acknowledgement
We have utilized code from several outstanding research works and sincerely thank the authors for their valuable discussions on the experiments, including those from [Vid2Avatar](https://github.com/MoyGcc/vid2avatar), [HumanNeRF](https://github.com/chungyiweng/humannerf), [SMPL-X](https://github.com/vchoutas/smplx), [MonoHuman](https://github.com/Yzmblog/MonoHuman), [PM-Avatar](https://github.com/ChunjinSong/pmavatar) and [NPC](https://github.com/LemonATsu/NPC-pytorch).
