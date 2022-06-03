# Orchid Classification

## 環境
```
pip install timm
pip install einops
pip install transformers
pip install ttach
pip install kornia
git clone https://github.com/travisergodic/orchid_classification.git
cd /content/orchid_classification
git clone https://github.com/davda54/sam.git
```

## 訓練
```
python train.py --config_file "config_v1.py"
```

## 評估
```
python test.py --model_paths "./checkpoints/model_v1.pt" --img_dir "./training"
```

## 預測
```
python predict.py --model_paths "./checkpoints/model_v1.pt" --target_dir "./training" --do_tta "True"
```
