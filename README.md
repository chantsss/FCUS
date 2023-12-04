# FCUS Official Repository

This repository is the official implementation of [FCUS: Traffic Rule-Aware Vehicle Trajectory Forecasting Using Continuous Unlikelihood and Signal Temporal Logic Feature](https://chantsss.github.io/FCUS/).

![Image text](https://github.com/chantsss/FCUS/docs/overview.png)


The code in this repository include the FCUS implementation based on the prediction generator backbone [AutoBots](https://arxiv.org/abs/2104.00563).

The code works well on NVIDIA GeForce RTX 3080 Ti NVIDIA-SMI 510.39.01    Driver Version: 510.39.01    CUDA Version: 11.6

### Env installment

1. Create a python 3.8.16 environment. I use Miniconda3 and create with 
`conda create --name fcus python=3.8.16`
2. Run `pip install -r requirements.txt`
3. Run `conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia`<br />
   or Run `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`, it depends on your device.

### nuScenes Installation and Setup

Download the dataset and the map expansion (v1.3) from [here](https://www.nuscenes.org/nuscenes#download).
Then follow the instructions [here](https://github.com/nutonomy/nuscenes-devkit) to install the nuscenes devkit.

Ensure that the final folder structure looks like this:
```
v1.0-trainval_full
 └──v1.0-trainval
      └──attribute.json
      └──calibrated_sensor.json
      └──category.json
      └──ego_pose.json
      └──instance.json
      └──log.json
      └──map.json
      └──sample.json
      └──sample_annotation.json
      └──sample_data.json
      └──scene.json
      └──sensor.json
      └──visibility.json
 └──maps
      └──basemap
            └──boston-seaport.png
            └──singapore-hollandvillage.png
            └──singapore-onenorth.png
            └──singapore-queenstown.png
      └──expansion
            └──boston-seaport.json
            └──singapore-hollandvillage.json
            └──singapore-onenorth.json
            └──singapore-queenstown.json
      └──prediction
          └──prediction_scenes.json
      └──36092f0b03a857c6a3403e25b4b7aab3.png
      └──37819e65e09e5547b8a3ceaefba56bb2.png
      └──53992ee3023e5494b90c316c183be829.png
      └──93406b464a165eaba6d9de76ca09f5da.png
      
```

### Data preprocess

Run the following to create the h5 files of the dataset with safe map:

```
python create_h5_nusc.py --raw-dataset-path /path/to/nuscenes_dataset/ --split-name [train/val] --output-h5-path /path/to/output/nuscenes_h5_file/
```

### Training

The trained models will be saved in `results/{Dataset}/{exp_name}`. 

Make sure you are using Autobot-Ego-Gan. 
You can turn on NLL and STL by adding --use-gan True --use-nll True. More specific training setting please refer to process_args.py. 

```
python train.py --exp-id test --seed 1 --dataset Nuscenes --model-type Autobot-Ego-Gan --num-modes 10 --hidden-size 128 --num-encoder-layers 2 --num-decoder-layers 2 --dropout 0.1 --entropy-weight 40.0 --kl-weight 20.0 --use-FDEADE-aux-loss True --use-map-lanes True --tx-hidden-size 384 --batch-size 64 --learning-rate 0.00075 --learning-rate-sched 10 20 30 40 50 --dataset-path /your/preprocessd/data/path/ --use-gan True --use-nll True --use-continuous True
```

### Evaluating

```
python evaluate.py --dataset-path /path/to/root/of/interaction_dataset_h5_files --models-path /your/model/path/{model_epoch}.pth --batch-size 64
```

### For submitting to the nuScenes

```
python useful_scripts/generate_nuscene_results.py --dataset-path /path/to/root/of/nuscenes_h5_files --models-path results/Nuscenes/{exp_name}/{model_epoch}.pth 
```


## Reference

If you find this repository is useful, please cite our work:

```
Will update as soon as possible.
```


