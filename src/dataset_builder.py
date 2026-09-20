import os
import pandas as pd
import numpy as np
from PIL import Image

import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split

from src.config import Config

#costruiamo un dataset unificato, poi eseguiamo lo split sulle immagini sorgente 
class MergedDatasetSplitter:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    #costruzione dataset 1: COCO + powerpaint
    def _build_ds1(self):
        #costruiamo il manifest per il dataset 1
        df = pd.read_csv(self.cfg.CSV_PATH) #leggiamo il csv originale del primo dataset
        rows, missing = [], []

        #per ogni riga del csv ricostruisce il percorso esatto in cui dovrebbero trovarsi l'immagine originale, quella manipolata e la maschera
        for _, r in df.iterrows():
            image_id = str(r["image_id"])
            orig_path = os.path.join(self.cfg.ORIG_DIR,  f"{image_id}.jpg")
            manip_path = os.path.join(self.cfg.MANIP_DIR, f"{image_id}.jpg")
            mask_path = os.path.join(self.cfg.MASK_DIR,  f"{image_id}_mask.png")

            #check per verificare che la tripla esista davvero
            for p in (orig_path, manip_path, mask_path):
                if not os.path.exists(p):
                    missing.append(p)

            rows.append({
                "image_id": f"ds1_{image_id}", #aggiungiamo un prefisso per evitare collisioni
                "orig_path": orig_path,
                "manip_path": manip_path,
                "mask_path": mask_path,
                "category_name": r.get("category_name", "unknown"),
                "target_prompt": r.get("target_prompt", ""),
                "source": "ds1",
                "mask_type": "png" #le maschere di coco sono in formato png
            })

        self._report_missing(missing, "Dataset 1")

        ds1 = self._remove_black_images(pd.DataFrame(rows)) #rimuoviamo le immagini danneggiate (problema di ppt che se classifica le immagini come NSFW le restituisce completamente nere)
        return ds1

    #costruzione dataset 2: bagel + HDpainter
    def _build_ds2(self):
        #costruiamo il manifest per il dataset 2
        df = pd.read_csv(self.cfg.DS2_CSV_PATH)
        rows, missing = [], []

        #la logica di costruzione del dataset è la stessa del primo, ma cambiano i nomi dei file perchè sono strutturati diversamente
        for _, r in df.iterrows():
            first_file = str(r["first_level_file"]) #es."0000_BAGEL.jpg"
            second_file = str(r["second_level_file"]) # es."0000_BAGEL-HDPAINTER.jpg"
            category = str(r["original_class"])
            prompt = str(r.get("prompt", ""))

            #ricostruzione path
            orig_path = os.path.join(self.cfg.DS2_ORIG_DIR,  first_file)
            manip_path = os.path.join(self.cfg.DS2_MANIP_DIR, second_file)

            #elaboriamo il nome di ogni maschera, la struttura è: {stem}_{original_class}.npy
            stem = os.path.splitext(first_file)[0]  #rimuove .jpg/.png
            mask_name = f"{stem}_{category}.npy"
            mask_path = os.path.join(self.cfg.DS2_MASK_DIR, mask_name)

            #check per le immagini mancanti
            for p in (orig_path, manip_path, mask_path):
                if not os.path.exists(p):
                    missing.append(p)

            #costruiamo l'image_id unico: prefisso ds2 + stem del file originale
            image_id = f"ds2_{stem}"

            rows.append({
                "image_id": image_id,
                "orig_path": orig_path,
                "manip_path": manip_path,
                "mask_path": mask_path,
                "category_name": category,
                "target_prompt": prompt,
                "source": "ds2",
                "mask_type": "npy" #le maschere di bagel sono in formato numpy
            })

        self._report_missing(missing, "Dataset 2")
        return pd.DataFrame(rows)


    def _split_by_image_id(self, manifest):        
        # Tabella di lookup image_id → source (ogni image_id ha una sola source)
        id_source = manifest.drop_duplicates("image_id")[["image_id", "source"]].set_index("image_id")["source"]
        unique_ids = id_source.index.to_numpy() #vettore con gli id unici
        strata = id_source.to_numpy() #vettore con la label del dataset di provenienza

        #split per ottenere il training set
        train_ids, rest_ids, _, rest_strata = train_test_split(
            unique_ids, strata,
            train_size=self.cfg.TRAIN_FRAC,
            random_state=self.cfg.SEED,
            stratify=strata #ci assicuriamo che la proporzione tra i due dataset in ogni split sia del 50%
        )

        #split delle immagini restanti tra validation e test
        relative_val = self.cfg.VAL_FRAC / (self.cfg.VAL_FRAC + self.cfg.TEST_FRAC)
        val_ids, test_ids = train_test_split(
            rest_ids,
            train_size=relative_val,
            random_state=self.cfg.SEED,
            stratify=rest_strata #snche in questo caso è proporzionato
        )

        id_to_split = (
            {i: "train" for i in train_ids} |
            {i: "val" for i in val_ids}   |
            {i: "test" for i in test_ids}
        ) #unifichiamo le liste

        manifest = manifest.copy()
        manifest["split"] = manifest["image_id"].map(id_to_split) #aggiungiamo una colonna al manifest con le informazioni sullo split
        return manifest

    #check di sicurezza per verificare che non ci sia data leakage (nessuna immagine appartenente al training set deve essere anche nel test set)
    def _verify_no_leakage(self, manifest):
        splits_per_id = manifest.groupby("image_id")["split"].nunique()
        leaking = splits_per_id[splits_per_id > 1]
        if len(leaking) > 0:
            raise ValueError(f"Leakage rilevato su {len(leaking)} image_id.")
        print("Verifica leakage: OK.")


    @staticmethod
    def _report_missing(missing: list, name: str):
        if missing:
            print(f"[{name}] ATTENZIONE: {len(missing)} file mancanti:")
            for m in missing[:5]:
                print(f"  - {m}")
            raise FileNotFoundError(f"[{name}] file mancanti.")


    @staticmethod
    def _remove_black_images(manifest: pd.DataFrame) -> pd.DataFrame:
        print("\n[PULIZIA] Rimozione immagini nere.")
        to_keep = []
        removed_images = []
        black_count = 0

        for idx, row in manifest.iterrows():
            manip_path = row["manip_path"]
            is_black = False

            try:
                #apre l'immagine, la converte in array e controlla i pixel
                with Image.open(manip_path) as img:
                    arr = np.array(img)
                    
                    # np.any() è True se c'è almeno un valore > 0.
                    # Se l'array è tutto zero, not np.any() sarà True.
                    if not np.any(arr):
                        is_black = True
            except Exception as e:
                print(f"Errore durante la lettura di {manip_path}: {e}")
                is_black = True  #scartiamo il file se è corrotto o illeggibile
                removed_images.append(manip_path)

            if is_black:
                black_count += 1
            else:
                to_keep.append(row)

        if black_count > 0:
            print(f"Rimosse {black_count} coppie (immagine manipolata completamente nera o illeggibile).")
            #per stamparle in colonna e fare delle verifiche manuali
            for element in removed_images:
                print(element)

        #ricostruisce il DataFrame mantenendo i tipi originali
        return pd.DataFrame(to_keep)


    def run(self):
        print("MERGE E SPLIT DEL DATASET")

        #costruzione del primo dataset
        print("\n[1/4] Costruzione manifest Dataset 1...")
        ds1 = self._build_ds1()
        print(f" - {len(ds1)} coppie (ds1)")

        #costruzione del secondo dataset
        print("\n[2/4] Costruzione manifest Dataset 2...")
        ds2 = self._build_ds2()
        print(f" - {len(ds2)} coppie (ds2)")

        #unione dei due dataset e split di quello complessivo
        print("\n[3/4] Merge e split...")
        manifest = pd.concat([ds1, ds2], ignore_index=True)

        print(manifest.columns.tolist())
        
        #facciamo un undersampling per fare in modo che i due dataset di partenza contengano lo stesso numero di immagini
        if self.cfg.DATASET_SIZE is not None:
            print(f"\n[UNDERSAMPLING] Riduzione a {self.cfg.DATASET_SIZE} campioni per source...")
            parts = []
            for source_name, group in manifest.groupby("source"):
                n = min(self.cfg.DATASET_SIZE, len(group)) # evito che la dimensione dell'undersampling sia maggiore dell'intero dataset, in questo modo ho semore qualcosa da prendere
                parts.append(group.sample(n=n, random_state=self.cfg.SEED))
            manifest = pd.concat(parts, ignore_index=True)
            print(f" - {len(manifest)} coppie totali dopo undersampling")

        dupes = manifest["image_id"].duplicated()
        #check dei duplicati
        if dupes.any():
            print(f"  ATTENZIONE: {dupes.sum()} image_id duplicati:")
            print(manifest[dupes][["image_id", "source"]].to_string())

        manifest = self._split_by_image_id(manifest)
        self._verify_no_leakage(manifest)

        print("\n[4/4] Distribuzione per split:")
        dist = manifest.groupby(["split", "source"]).size().unstack(fill_value=0)
        print(dist)
        print("\nProporzione ds2/(ds1+ds2) per split:")
        print((dist["ds2"] / dist.sum(axis=1)).round(3))

        os.makedirs(os.path.dirname(self.cfg.MANIFEST_OUT), exist_ok=True)
        manifest.to_csv(self.cfg.MANIFEST_OUT, index=False)
        print(f"\nManifest salvato in: {self.cfg.MANIFEST_OUT}")

        return manifest
