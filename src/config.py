# ==============================================================================
# 0. CONFIGURAZIONE E RIPRODUCIBILITÀ
# ==============================================================================
import torch
from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent # punta all directory base

    # -- TEST ---
    DATA_DIR = None

    # --- Dataset 1 ---
    CSV_PATH = f"{BASE_DIR}/dataset/COCO/dataset.csv"
    ORIG_DIR = f"{BASE_DIR}/dataset/COCO/0" # Originali
    MANIP_DIR = f"{BASE_DIR}/dataset/COCO/1" # Manipolate
    MASK_DIR = f"{BASE_DIR}/dataset/COCO/masks"


   # --- Dataset 2 --- NONE sul csv path per non utilizzarlo
    DS2_CSV_PATH  = f"{BASE_DIR}/dataset/Bagel/dataset.csv"
    DS2_ORIG_DIR  = f"{BASE_DIR}/dataset/Bagel/0" # Originali
    DS2_MANIP_DIR = f"{BASE_DIR}/dataset/Bagel/1" # Manipolate
    DS2_MASK_DIR  = f"{BASE_DIR}/dataset/Bagel/masks"


    # --- Percorsi output ---
    MANIFEST_OUT = f"/{BASE_DIR}/dataset/manifest_split.csv"
    CHECKPOINT_DIR = f"{BASE_DIR}/outputs/checkpoints"
    FIGURES_DIR = f"{BASE_DIR}/outputs/figures"


    # --- Undersampling ----
    # None per non eseguirlo, il seed di sampling utilizzato è su varie
    DATASET_SIZE = 4000


    # --- Split ---
    TRAIN_FRAC = 0.70
    VAL_FRAC = 0.20
    TEST_FRAC = 0.10


    # --- Preprocessing ---
    IMAGE_SIZE = 256


    # --- Training ---
    BATCH_SIZE = 16
    EPOCHS = 60
    LR = 1e-4 # LR iniziale
    BASE_CHANNELS = 64
    NUM_WORKERS = 0
    

    # --- Early Stopping ---
    ES_PATIENCE  = 14    # usiamo un numeri alti in quanto con il cosine il lr scende lentamente
    ES_MIN_DELTA = 1e-3 # miglioramento minimo considerato significativo


    # --- Postprocessing ---
    SEG_THRESHOLD = 0.15 # Threshold di segmentazione, usato per la IoU
    CLS_THRESHOLD = 0.5 # Threshold di classificazione binaria 0.5


    # --- Varie ---
    SEED = 42 # Influenza tutto ciò che ha bisogno di un seed
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def set_new_data(self, new_data_dir, csv_name):
        import os
        self.DATA_DIR = new_data_dir
        self.CSV_PATH = os.path.join(new_data_dir, csv_name)

    def debug_paths(self):
        print(self.DATA_DIR)
        print(self.CSV_PATH)

    def set_undersampling(self, limit):
        self.DATASET_SIZE = limit

    def set_num_workers(self, num_workers):
        self.NUM_WORKERS = num_workers