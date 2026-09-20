# pyright: reportAssignmentType=false

import os
import numpy as np
import random
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

#applichiamo le trasformazioni a immagine e maschera simultaneamente
class JointTransform:
    def __init__(self, size, train, aggressive_mode = True):
        self.size = size
        self.train = train
        self.aggressive_mode = aggressive_mode

    def __call__(self, image: Image.Image, mask: Image.Image):
        #resize di base applicato sempre
        image = TF.resize(image, (self.size, self.size), interpolation=TF.InterpolationMode.BILINEAR)
        mask  = TF.resize(mask,  (self.size, self.size), interpolation=TF.InterpolationMode.NEAREST)

        if self.train:
            if self.aggressive_mode:
                #applichiamo sempre questa modalità di data aumentation al training set

                #colore
                image = TF.adjust_hue(image, random.uniform(-0.1, 0.1))
                image = TF.adjust_saturation(image, random.uniform(0.7, 1.3))

                #grayscale casuale
                if random.random() > 0.8:
                    image = TF.rgb_to_grayscale(image, num_output_channels=3)

                #multi-scala: distrugge le alte frequenze e forza la rete a ragionare su strutture a bassa frequenza
                if random.random() > 0.5:
                    small_size = self.size // 2  #da 256 a 128
                    image = TF.resize(image, (small_size, small_size), interpolation=TF.InterpolationMode.BILINEAR)
                    image = TF.resize(image, (self.size,  self.size), interpolation=TF.InterpolationMode.BILINEAR)
                    mask  = TF.resize(mask,  (small_size, small_size), interpolation=TF.InterpolationMode.NEAREST)
                    mask  = TF.resize(mask,  (self.size,  self.size), interpolation=TF.InterpolationMode.NEAREST)

            else:
                #data augmentation standard, l'abbiamo usato solo nei test ma non è risultato efficace 
                if random.random() > 0.5:
                    image, mask = TF.hflip(image), TF.hflip(mask)
                if random.random() > 0.5:
                    image, mask = TF.vflip(image), TF.vflip(mask)

                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle)
                mask  = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
                image = TF.adjust_contrast(image,   random.uniform(0.8, 1.2))

        #applichiamo sempre la normalizzazione
        image = TF.to_tensor(image)
        image = (image - 0.5) / 0.5

        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()

        return image, mask


#usiamo questa classe per recuperare i file dal disco quando sono necessari, partendo dal manifest
class InpaintingDataset(Dataset):
    def __init__(self, manifest_df, split, transform: JointTransform):
        self.rows = manifest_df[manifest_df["split"] == split].reset_index(drop=True) #manteniamo solo lo split che ci interessa e riscriviamo gli indici
        self.transform = transform

    def __len__(self):
        return len(self.rows) * 2 #perchè per ogni immagine abbiamo sia la versione originale che quella manipolata

    def __getitem__(self, idx):
        row = self.rows.iloc[idx // 2] #recuperiamo l'indice
        is_manipulated = (idx % 2 == 1) #se è dispari corrisponde a un'immagine manipolata, se è pari a una originale

        if is_manipulated:
            #se è manipolata apre direttamente il file
            image = Image.open(row["manip_path"]).convert("RGB")
            
            ext = os.path.splitext(row["mask_path"])[1]
            #controlla l'estensione della maschera (se è numpy la converte sennò la carica direttamente)
            if ext == ".npy":
                mask = Image.fromarray((np.load(row["mask_path"])*255).astype(np.uint8)).convert('L')
            else:
                mask = Image.open(row["mask_path"]).convert("L")
            
            label = 1.0 #assegna l'etichetta
        else:
            #se è originale apre l'immagine e crea una maschera completamente nera sul momento
            image = Image.open(row["orig_path"]).convert("RGB")
            mask = Image.new("L", image.size, 0)  # maschera nera generata al volo
            label = 0.0

        image, mask = self.transform(image, mask) #passa l'immagine per il preprocessing e eventuale data augmentation
        return image, mask, torch.tensor(label, dtype=torch.float32)
