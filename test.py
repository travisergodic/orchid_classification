import torch
import os
import pandas as pd 
import argparse
from evaluate import Evaluator
from data import build_test_transform
from configs.test_config import *


def boolean_string(s):
    if s == 'False': 
        return False
    elif s == 'True': 
        return True   
    else:
        raise ValueError('Not a valid boolean string')

def evaluate_all(model_paths, test_image_path_list, test_label_list):
    # load models
    models = [torch.load(model_path.strip()) for model_path in model_paths.split(",")]

    # evaluate
    test_image_transforms =  [build_test_transform(img_size) for img_size in test_img_size_list]
    evaluator = Evaluator(models, test_image_transforms, device=DEVICE, activation=activation)
    ## no TTA
    score = evaluator.evaluate(test_image_path_list, test_label_list, metric, False)
    print(f"No TTA: {score} ({metric} score).")
    ## TTA
    score = evaluator.evaluate(test_image_path_list, test_label_list, metric, tta_fn)
    print(f"With TTA: {score} ({metric} score).")   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image classification for flowers!")
    parser.add_argument("--model_paths", type=str)
    parser.add_argument("--img_dir", type=str)
    args = parser.parse_args()

    # read test file
    df_test_label = pd.read_csv('./Labels/test_label.csv')
    test_image_path_list = [os.path.join(args.img_dir, filename) for filename in list(df_test_label['filename'])] 
    test_label_list = list(df_test_label['category'])

    # build model
    evaluate_all(args.model_paths, test_image_path_list, test_label_list)