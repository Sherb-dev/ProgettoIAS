import os
import glob
import pandas as pd
from src.config import Config

class SagiDatasetBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _build_dataset(self):
        df = pd.read_csv(self.cfg.CSV_PATH)
        rows = []
        missing = []

        # Filtro opzionale se il CSV ha colonne esplicite per dataset e method
        if 'dataset' in df.columns and 'method' in df.columns:
            df_filtered = df[(df['dataset'] == 'coco') & (df['method'] == 'hdpainter')]
        else:
            df_filtered = df

        for _, r in df_filtered.iterrows():
            filename = str(r.get("filename", r.get("image_id", "")))
            stem = os.path.splitext(filename)[0]
            
            # Lo split (train/val/test) in cui si trova l'immagine 
            split_folder = str(r.get("split", "train"))

            # Ricostruzione percorsi con wildcard
            orig_path = os.path.join(self.cfg.DATA_DIR, split_folder, "coco", "original", f"{stem}.jpg")
            mask_pattern = os.path.join(self.cfg.DATA_DIR, split_folder, "coco", "mask", f"coco_{stem}_*.png")
            manip_pattern = os.path.join(self.cfg.DATA_DIR, split_folder, "coco", "hdpainter", f"coco_{stem}_*.png")

            # Ricerca dei file corrispondenti ai pattern
            mask_matches = glob.glob(mask_pattern)
            manip_matches = glob.glob(manip_pattern)

            mask_path = mask_matches[0] if mask_matches else ""
            manip_path = manip_matches[0] if manip_matches else ""

            # Tracciamento file mancanti
            if not os.path.exists(orig_path): missing.append(orig_path)
            if not mask_path: missing.append(mask_pattern)
            if not manip_path: missing.append(manip_pattern)

            # Costruzione della riga solo se la tripla è completa, 
            # rispettando esattamente il vecchio header
            if os.path.exists(orig_path) and mask_path and manip_path:
                rows.append({
                    "image_id": f"sagi_coco_{stem}", 
                    "orig_path": orig_path,
                    "manip_path": manip_path,
                    "mask_path": mask_path,
                    "category_name": r.get("category_name", "unknown"),
                    "target_prompt": r.get("target_prompt", ""),
                    "source": "sagi_hdpainter", 
                    "mask_type": "png",
                    "split": split_folder  # Necessario per mantenere il partizionamento a valle
                })

        if missing:
            print(f"ATTENZIONE: {len(missing)} file/pattern mancanti (ne mostro 5):")
            for m in list(set(missing))[:5]:
                print(f"  - {m}")

        return pd.DataFrame(rows)

    def _undersample(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.cfg.DATASET_SIZE is None or self.cfg.DATASET_SIZE >= len(df):
            return df

        print(f"Esecuzione undersampling stratificato a {self.cfg.DATASET_SIZE} campioni...")
        
        # Calcolo le proporzioni attuali degli split nel dataset per mantenerle inalterate
        split_counts = df["split"].value_counts(normalize=True)
        sampled_parts = []
        
        for split_name, proportion in split_counts.items():
            n_samples = int(self.cfg.DATASET_SIZE * proportion)
            split_group = df[df["split"] == split_name]
            
            n = min(n_samples, len(split_group))
            sampled_parts.append(split_group.sample(n=n, random_state=self.cfg.SEED))
            
        df_sampled = pd.concat(sampled_parts, ignore_index=True)
        
        # Correzione per eventuali discrepanze dovute agli arrotondamenti (es. 99 campioni invece di 100)
        diff = self.cfg.DATASET_SIZE - len(df_sampled)
        if diff > 0:
            remaining = df.drop(df_sampled.index)
            if not remaining.empty:
                extra = remaining.sample(n=min(diff, len(remaining)), random_state=self.cfg.SEED)
                df_sampled = pd.concat([df_sampled, extra], ignore_index=True)

        return df_sampled

    def run(self):
        print("AVVIO DATASET BUILDER (SAGI-D COCO + HD-Painter)")
        manifest = self._build_dataset()
        print(f"Trovate {len(manifest)} coppie valide prima dell'undersampling.")

        # Esecuzione undersampling
        manifest = self._undersample(manifest)
        print(f"Totale campioni finali: {len(manifest)}")

        # Riordino esplicito per forzare la struttura esatta dell'header richiesto
        expected_columns = [
            "image_id", "orig_path", "manip_path", "mask_path", 
            "category_name", "target_prompt", "source", "mask_type", "split"
        ]
        
        final_columns = [col for col in expected_columns if col in manifest.columns]
        manifest = manifest[final_columns]

        # Salvataggio
        os.makedirs(os.path.dirname(self.cfg.MANIFEST_OUT), exist_ok=True)
        manifest.to_csv(self.cfg.MANIFEST_OUT, index=False)
        print(f"Manifest salvato con successo in: {self.cfg.MANIFEST_OUT}")

        return manifest