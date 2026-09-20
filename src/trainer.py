import os
import random
 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
 
from src.loss import *
from src.config import Config
 
#implementiamo a mano l'early stopping perchè non esiste nativamente come in keras
#fermiamo il training se la loss in validazione non migliora per n epoche
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta #miglioramento minimo richiesto
        self.counter = 0
        self.best_loss = float("inf") #inizializziamo la best loss a +infinito in modo che ci sia a prescindere un miglioramento nella prima epoca
        self.should_stop = False
 
    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1 #se la val_loss non migliora aumentiamo il contatore
            if self.counter >= self.patience:
                self.should_stop = True #se per n epoche non c'è stato un miglioramento fermiamo il modello
        return self.should_stop
 
 
class Trainer:
    def __init__(self, model, cfg, model_name):
        self.model = model.to(cfg.DEVICE)
        self.cfg = cfg
        self.model_name = model_name
 
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=cfg.LR, weight_decay=1e-4
        ) #usiamo l'ottimizzatore Adam weight decay, che penalizza i pesi troppo grandi per evitare l'overfitting
 
        #usiamo un doppio scheduler per modificare il learning rate durante l'addestramento
        self.scheduler = optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[
                optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=5
                ), #primo scheduler di warmup con lr che aumenta gradualmente
                optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=cfg.EPOCHS - 5, eta_min=1e-6
                ), #dalla quinta epoca in poi il lr diminuisce gradualmente seguendo la curva del coseno per evitare sbalzi del gradiente
            ],
            milestones=[5]
        )
 
        #early stopping
        self.early_stopping = EarlyStopping(
            patience=cfg.ES_PATIENCE,
            min_delta=cfg.ES_MIN_DELTA
        )
 
        self.history = {"train_loss": [], "val_loss": []}
        self.best_val_loss = float("inf")
 
    def _run_epoch(self, loader: DataLoader, train, desc):
        self.model.train() if train else self.model.eval() #modalità
        total_loss = 0.0
 
        context = torch.enable_grad() if train else torch.no_grad() #aggiorniamo i pesi (gradienti attivi) solo se train=True
        with context:
            for image, mask, label in tqdm(loader, desc=desc, mininterval=10.0): #il loader fornisce un batch alla volta
                #trasferiamo le immagini dalla cpu alla gpu
                image = image.to(self.cfg.DEVICE)
                mask  = mask.to(self.cfg.DEVICE)
                label = label.to(self.cfg.DEVICE)
 
                if train:
                    self.optimizer.zero_grad() #per ogni batch ripuliamo i gradienti per evitare che si sommino
 
                seg_logits, cls_logits = self.model(image) #output di segmentazione e classificazione della rete
                loss = combined_loss(seg_logits, mask, cls_logits, label) #calcoliamo la loss sul batch
 
                #backpropagation
                if train:
                    loss.backward()
                    self.optimizer.step()
               
                #loss totale del batch corrente
                total_loss += loss.item() * image.size(0)
 
        return total_loss / len(loader.dataset) #restituisce l'errore medio dell'epoca
 
 
    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        os.makedirs(self.cfg.CHECKPOINT_DIR, exist_ok=True)
        model_name = self.model_name
 
        for epoch in range(self.cfg.EPOCHS):
            #per ogni epoca addestriamo e validiamo i risultati
            train_loss = self._run_epoch(
                train_loader, train=True,
                desc=f"[{model_name}] Epoch {epoch+1}/{self.cfg.EPOCHS} train"
            )
            val_loss = self._run_epoch(
                val_loader, train=False,
                desc=f"[{model_name}] Epoch {epoch+1}/{self.cfg.EPOCHS} val"
            )
 
            self.scheduler.step() #aggiorniamo il learning rate
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
 
            #stampiamo le metriche dell'epoca corrente
            print(
                f"[{model_name}] Epoch {epoch+1}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f} | "
                f"ES counter={self.early_stopping.counter}/{self.early_stopping.patience}"
            )
 
            #salviamo il miglior checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                ckpt_path = os.path.join(
                    self.cfg.CHECKPOINT_DIR, f"{model_name}_best.pt"
                )
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"Checkpoint salvato (val_loss={val_loss:.4f})")
 
            #early stopping
            if self.early_stopping.step(val_loss):
                print(
                    f"\n[Early Stopping] Nessun miglioramento per "
                    f"{self.early_stopping.patience} epoche. "
                    f"Training fermato all'epoca {epoch+1}."
                )
                break
 
        #ripristiniamo il modello migliore con il checkpoint
        best_ckpt = os.path.join(self.cfg.CHECKPOINT_DIR, f"{model_name}_best.pt")
        self.model.load_state_dict(torch.load(best_ckpt, map_location=self.cfg.DEVICE))
        print(f"[Trainer] Miglior modello ripristinato da: {best_ckpt}")
 
        #salviamo la history per l'analisi e i plot
        history_df   = pd.DataFrame(self.history)
        history_path = os.path.join(
            self.cfg.CHECKPOINT_DIR, f"{model_name}_history.csv"
        )
        history_df.to_csv(history_path, index_label="epoch")
        print(f"History salvata in: {history_path}")
 
        return self.history