import os
import random
 
import numpy as np
import torch
 
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import torch.nn as nn
 
# Custom Imports
from src.loss import *
from src.config import Config
from src.trainer import Trainer
from src.unet import AttentionUNet, PlainUNet
from src.evaluator import Evaluator
from src.dataset_builder import MergedDatasetSplitter
from src.preprocessing import JointTransform, InpaintingDataset
 
 
"""
    ====================================================================================
                                            TRAIN
    ====================================================================================
    QUESTO FILE LO ABBIAMO UTILIZZATO SOLO PER ESEGUIRE L'ADDESTRAMENTO TRAMITE SCREEN,
     QUESTO CODICE E TUTTO IL RESTO VIENE DOCUMENTATO MEGLIO ALL'INTERNO DEL NOTEBOOK:
                                        relazione.ipynb
"""
 
def set_seed(seed):
    #fissiamo i seed per rendere gli esperimenti riproducibili
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 
 
def main(model_name, model_fn):
    # model_fn è l'architetture, la passiamo come costruttore e non come istanza già creata
    cfg = Config()
    set_seed(cfg.SEED)
 
    model = model_fn() #istanziamo qui il modello, dopo set_seed (importante per la riproducibilità)
 
    # creiamo il manifest del dataset
    manifest = MergedDatasetSplitter(cfg).run() #split del dataset e creazione del manifest
 
    # trasformiamo le liste in oggetti dataset, ridimensioniamo immagini e maschere e nel caso di train=True facciamo data augmentation
    train_ds = InpaintingDataset(manifest, "train", JointTransform(cfg.IMAGE_SIZE, train=True))
    val_ds = InpaintingDataset(manifest, "val", JointTransform(cfg.IMAGE_SIZE, train=False))
    test_ds = InpaintingDataset(manifest, "test", JointTransform(cfg.IMAGE_SIZE, train=False))
 
    # suddivisione in batch con shuffle solo nel training set
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
 
    # training (stessa procedura per entrambe le reti)
    Trainer(model=model, cfg=cfg, model_name=model_name).fit(train_loader, val_loader)
    
    # salviamo i checkpoint in caso di miglioramenti
    # a fine addestramento ripristiniamo lo stato miglio della rete in modo da valutare la versione ottimale
    model.load_state_dict(torch.load(os.path.join(cfg.CHECKPOINT_DIR, f"{model_name}_best.pt")))
 
    # valutazione
    results = Evaluator(model, cfg).evaluate(test_loader)
 
    metrics={k: v for k, v in results.items() if k not in ("fpr", "tpr", "labels", "scores")}
 
    print(f"\n Risultati per {model_name}:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
   
    print(f"\n[{model_name}]: pipeline completata")
 

if __name__ == "__main__":
    main(model_name="UNet_Plain_2", model_fn=PlainUNet)
    main(model_name="UNet_Att_2", model_fn=AttentionUNet)