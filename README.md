# Cluster

## Datasets
To train on the MVtec Anomaly Detection dataset [download](https://www.mvtec.com/company/research/datasets/mvtec-ad)
the data and extract it. The [Describable Textures dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/) was used as the anomaly source 
image set in most of the experiments in the paper. You can run the **download_dataset.sh** script from the project directory
to download the MVTec and the DTD datasets to the **datasets** folder in the project directory:
```
./scripts/download_dataset.sh
```


## Training
Pass the folder containing the training dataset to the **train_DRAEM.py** script as the --data_path argument and the
folder locating the anomaly source images as the --anomaly_source_path argument. 
The training script also requires the batch size (--bs), learning rate (--lr), epochs (--epochs), path to store checkpoints
(--checkpoint_path), path to store logs (--log_path), the number of cluster centers (--k) and the dimensionality of each cluster center (--center_dim).
Example:

```
python train_DRAEM.py --gpu_id 0 --obj_id -1 --lr 0.0001 --bs 4 --epochs 800 --data_path ../mvtec_anomaly_detection/ --anomaly_source_path ../dtd/images/ --checkpoint_path ./checkpoints/ --log_path ./logs/ --visualize True --k 10 --center_dim 256
```

The conda environement used in the project is decsribed in **requirements.txt**.

## Pretrained models
Pretrained Cluster models for each class of the MVTec anomaly detection dataset are available. 
Due to the large size of the parameter files, please contact me if you are interested in obtaining them.

The pretrained models achieve a 98.8 image-level ROC AUC and a 97.8 pixel-wise ROC AUC.


## Evaluating
The test script requires the --gpu_id arguments, the name of the checkpoint files (--base_model_name) for trained models, the 
location of the MVTec anomaly detection dataset (--data_path), the folder where the checkpoint files are located (--checkpoint_path), the number of cluster centers (--k) 
and the dimensionality of each cluster center (--center_dim) with pretrained models can be run with:

```
python test_DRAEM.py --gpu_id 0 --base_model_name "DRAEM_test_0.0001_800_bs4" --data_path ../mvtec_anomaly_detection/ --checkpoint_path ./checkpoints/ --k 10 --center_dim 256
```


