# Orchid Classification

## 前期準備
```
git clone https://github.com/travisergodic/orchid_classification.git
cd "./orchid_classification"
pip install -r requirements.txt
git clone https://github.com/rwightman/pytorch-image-models
cd pytorch-image-models 
pip install -e .
cd "../"
git clone https://github.com/davda54/sam.git
mkdir "./checkpoints"
mkdir "./final_models"
mkdir "./pred_data"
```

## 檔案下載
1. **下載資料**: 
   + 下載 `training.zip`, `orchid_private_set.zip`, `orchid_public_set.zip`：https://drive.google.com/drive/folders/1tWxR5XxRWJaWEz0cy8jMacHRWdmqa7Z1
   + 解壓縮 `training.zip` 檔案，並將壓縮後的資料夾 `training` 放在 `orchid_classification/` 路徑
   + 解壓縮 `orchid_private_set.zip`, `orchid_public_set.zip` 檔案，並將解壓後的所有圖片放在 `orchid_classification/pred_data/` 路徑    

2. **下載比賽所使用模型**
   + 下載路徑：https://drive.google.com/drive/folders/1__OQRk6SlWDfsqvstSi2gSItakveAxcb
   + 將 `convnext_v8.pt`, `convnext_v10.pt` 放在 `orchid_classification/final_models/` 路徑

## 使用方法
   1. **訓練**
   ```
   python train.py --config_file "convnext/config_v8.py"
   ```

   2. **預測**
   ```
   python predict.py --model_paths "./final_models/convnext_v10.pt" \
                   --target_dir "/content/orchid_classification/pred_data" \
                   --do_tta "False"
   ```
