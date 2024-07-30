# Deep learning for the identification of nanopore protein sequencing 

### Installing Dependencings
Python version > 3.9 was used in this project. Enviroment can be created by ``` conda create -n ps39 python=3.9 ```. After activate the enviroment ```conda activate ps39```, the required packages can be installed by: 
```
pip install -r requirements.txt
```
Install torch
```
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
```

### Training 
The CNN network for peptide identification can be trained as:
```
python train_feat.py --lr 0.0001 --train_pkl_file ./data/data_train.pkl --val_pkl_file ./data/data_val.pkl --exp_name peptide_cnn --padding 100
```

### Test 
The CNN network for peptide identification can be trained as:
```
python test_feat.py --hp_pkl_file ./data/data_test.pkl --padding 100 --checkpoint_file ./logs/dir_to_model/model_best.pth
```




