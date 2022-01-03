# NeRFlow
[ICCV'21] Neural Radiance Flow for 4D View Synthesis and Video Processing

### Datasets

The pouring dataset used for experiments can be download [at this URL](https://www.dropbox.com/s/bnnjixu1ihyxwn3/pouring_dataset.tar.gz?dl=0) and the iGibson dataset used in experiments can be downloaded [at this URL](https://www.dropbox.com/s/iu12rz0emjp5ija/gibson_dataset.tar?dl=0)

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

### Citing our Paper

If you find this repo helpful, please consider citing

```	
 @inproceedings{du2021nerflow,
                      author    = {Yilun Du and Yinan Zhang and Hong-Xing Yu 
                                   and Joshua B. Tenenbaum and Jiajun Wu},
                      title     = {Neural Radiance Flow for 4D View Synthesis and Video Processing},
                      year      = {2021},
                      booktitle   = {Proceedings of the IEEE/CVF International Conference
                                     on Computer Vision},
                    }
```	
