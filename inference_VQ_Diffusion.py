# ------------------------------------------
# VQ-Diffusion
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Shuyang Gu
# ------------------------------------------

import os
import sys
import shutil
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import torch
import cv2
import argparse
import numpy as np
import torchvision
from PIL import Image

from image_synthesis.utils.io import load_yaml_config
from image_synthesis.modeling.build import build_model
from image_synthesis.utils.misc import get_model_parameters_info
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--text', type=str, help='Tekstowy opis obrazu do wygenerowania')
parser.add_argument('--truncation_rate', type=float, default=0.85, help='Współczynnik obcięcia próbkowania')
parser.add_argument('--batch_size', type=int, default=4, help='Liczba obrazów w batchu')
parser.add_argument('--fast', type=int, default=2, help='Parametr przyspieszenia (od 2 do 10)')
parser.add_argument('--config', type=str, default='/kaggle/working/VQ-Diffusion/configs/coco.yaml', help='Ścieżka do configu')
parser.add_argument('--weights', type=str, default='/kaggle/working/coco_pretrained.pth', help='Ścieżka do wag modelu')
parser.add_argument('--save_root', type=str, default='RESULT', help='Folder zapisu wyników')

args = parser.parse_args()


class VQ_Diffusion():
    def __init__(self, config, path):
        self.info = self.get_model(ema=True, model_path=path, config_path=config)
        self.model = self.info['model']
        self.epoch = self.info['epoch']
        self.model_name = self.info['model_name']
        self.model = self.model.cuda()
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad=False

    def get_model(self, ema, model_path, config_path):
        import os
        print("Current working dir:", config_path)
        if 'OUTPUT' in model_path: # pretrained model
            model_name = model_path.split(os.path.sep)[-3]
        else:
            model_name = os.path.basename(config_path).replace('.yaml', '')
        print(model_path)
        config = load_yaml_config(config_path)
        model = build_model(config)
        model_parameters = get_model_parameters_info(model)
        
        print(model_parameters)
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location="cpu")

        if 'last_epoch' in ckpt:
            epoch = ckpt['last_epoch']
        elif 'epoch' in ckpt:
            epoch = ckpt['epoch']
        else:
            epoch = 0

        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print('Model missing keys:\n', missing)
        print('Model unexpected keys:\n', unexpected)

        if ema==True and 'ema' in ckpt:
            print("Evaluate EMA model")
            ema_model = model.get_ema_model()
            missing, unexpected = ema_model.load_state_dict(ckpt['ema'], strict=False)
        
        return {'model': model, 'epoch': epoch, 'model_name': model_name, 'parameter': model_parameters}

    def inference_generate_sample_with_class(self, text, truncation_rate, save_root, batch_size,fast=False):
        os.makedirs(save_root, exist_ok=True)

        data_i = {}
        data_i['label'] = [text]
        data_i['image'] = None
        condition = text

        str_cond = str(condition)
        save_root_ = os.path.join(save_root, str_cond)
        os.makedirs(save_root_, exist_ok=True)

        with torch.no_grad():
            model_out = self.model.generate_content(
                batch=data_i,
                filter_ratio=0,
                replicate=batch_size,
                content_ratio=1,
                return_att_weight=False,
                sample_type="top"+str(truncation_rate)+'r',
            ) # B x C x H x W

        # save results
        content = model_out['content']
        content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
        for b in range(content.shape[0]):
            cnt = b
            save_base_name = '{}'.format(str(cnt).zfill(6))
            save_path = os.path.join(save_root_, save_base_name+'.jpg')
            im = Image.fromarray(content[b])
            im.save(save_path)

    def inference_generate_sample_with_condition(self, text, truncation_rate, save_root, batch_size,fast=False):
        os.makedirs(save_root, exist_ok=True)

        data_i = {}
        data_i['text'] = [text]
        data_i['image'] = None
        condition = text

        str_cond = str(condition)
        save_root_ = os.path.join(save_root, str_cond)
        os.makedirs(save_root_, exist_ok=True)

        if fast != False:
            add_string = 'r,fast'+str(fast-1)
        else:
            add_string = 'r'
        with torch.no_grad():
            model_out = self.model.generate_content(
                batch=data_i,
                filter_ratio=0,
                replicate=batch_size,
                content_ratio=1,
                return_att_weight=False,
                sample_type="top"+str(truncation_rate)+add_string,
            ) # B x C x H x W

        # save results
        content = model_out['content']
        content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
        for b in range(content.shape[0]):
            cnt = b
            save_base_name = '{}'.format(str(cnt).zfill(6))
            save_path = os.path.join(save_root_, save_base_name+'.png')
            im = Image.fromarray(content[b])
            im.save(save_path)


if __name__ == '__main__':
    captions_path = "/kaggle/working/fixed_captions.txt"
    ids_path = "/kaggle/working/fixed_image_ids.txt"
    output_dir = "/kaggle/working/output"

    os.makedirs(output_dir, exist_ok=True)

    # Wczytanie opisów i ID
    with open(captions_path, "r", encoding="utf-8") as f:
        captions = [line.strip() for line in f if line.strip()]

    with open(ids_path, "r", encoding="utf-8") as f:
        image_ids = [line.strip() for line in f if line.strip()]

    if len(captions) != len(image_ids):
        raise ValueError("Liczba opisów i ID nie jest taka sama!")

    # Ładowanie modelu
    VQ_Diffusion_model = VQ_Diffusion(
        config='/kaggle/working/VQ-Diffusion/configs/coco.yaml',
        path='/kaggle/working/coco_pretrained.pth'
    )

    for idx, (caption, img_id) in enumerate(zip(captions, image_ids)):
        if idx < 1344:
          continue
        print(f"[{idx+1}/{len(captions)}] Generuję obrazy dla: '{caption}'")

        # Ścieżka do folderu wynikowego
        result_dir = f"/kaggle/working/VQ-Diffusion/RESULT/{caption}"

        # Generowanie 5 obrazów
        VQ_Diffusion_model.inference_generate_sample_with_condition(
            caption,
            truncation_rate=0.85,
            save_root="/kaggle/working/VQ-Diffusion/RESULT",
            batch_size=5,
            fast=2
        )

        # Przenoszenie i zmiana nazw
        if os.path.exists(result_dir):
            files = sorted(os.listdir(result_dir))
            for img_num, filename in enumerate(files):
                src_path = os.path.join(result_dir, filename)
                dst_filename = f"generated_{img_id}_{img_num}.png"
                dst_path = os.path.join(output_dir, dst_filename)
                shutil.move(src_path, dst_path)

            # Usunięcie pustego folderu po przeniesieniu
            shutil.rmtree(result_dir)
        else:
            print(f"⚠ Brak folderu wynikowego dla '{caption}'")
