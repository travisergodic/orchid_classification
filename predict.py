import os
import argparse
import torch
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

def make_prediction(model_paths, image_dir, do_tta):
    # load models
    models = [torch.load(model_path).strip() for model_path in model_paths.split(",")]

    test_image_transforms =  [build_test_transform(img_size) for img_size in test_img_size_list]
    evaluator = Evaluator(models, 
                          test_image_transforms, 
                          device=DEVICE, 
                          activation=activation)

    evaluator.make_prediction(image_dir, tta_fn if do_tta else False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image classification for flowers!")
    parser.add_argument("--model_paths", type=str)
    parser.add_argument("--target_dir", type=str)
    parser.add_argument("--do_tta", type=boolean_string)
    args = parser.parse_args()
    
    # evaluate
    make_prediction(args.model_paths, args.target_dir, args.do_tta)