# NeRFlow
[ICCV'21] Neural Radiance for 4D View Synthesis and Video Processing

### Datasets

The pouring dataset used for experiments can be download [here]("") and the iGibson dataset used in 
experiments can be downloaded [here]("")

### Pouring Dataset

Please download and extract each dataset at data/nerf\_synthetic/. Please use the following command to train 

```
python run_nerf.py --config=configs/pour_baseline.txt
```

After running model for 200,000 iterations, move the model to a new folder pour\_dataset\_flow and then use the following command
to train with flow consistency

```
python run_nerf.py --config=configs/pour_baseline_flow.txt
```


### Gibson Dataset

Please download and extract each dataset at data/nerf\_synthetic/. Please use the following command to train the model 

```
python run_nerf.py --config=configs/gibson_baseline.txt
```

After running model for 200,000 iterations, move the model to a new folder pour\_dataset\_flow and then use the following command
to train with flow consistency

```
python run_nerf.py --config=configs/gibson_baseline_flow.txt
```
