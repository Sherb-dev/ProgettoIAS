import os
import pandas as pd
from src.config import Config

class SagiDatasetBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _build_dataset(self):
        df = pd.read_csv(self.cfg.CSV_PATH)
        rows = []
        missing = []


        # Filtriamo per hdpainter
        df_filtered = df[df["inpainting_model"] == 'hdpainter']


        # Filtriamo per assicurarci di prendere solo il dataset COCO
        df_filtered = df_filtered[df_filtered['src_path'].str.contains('coco', na=False)]

        for _, r in df_filtered.iterrows():
            # Il CSV presenta percorsi con backslash tipici di Windows (es. sagid\train\...)
            # Sostituiamo i backslash con il separatore del sistema operativo corrente
            src_rel = str(r["src_path"]).replace("sagid\\", "", 1).replace("\\", os.sep)
            img_rel = str(r["img_path"]).replace("sagid\\", "", 1).replace("\\", os.sep)
            mask_rel = str(r["mask_path"]).replace("sagid\\", "", 1).replace("\\", os.sep)

            # Costruzione dei percorsi assoluti
            orig_path = os.path.join(self.cfg.DATA_DIR, src_rel)
            manip_path = os.path.join(self.cfg.DATA_DIR, img_rel)
            mask_path = os.path.join(self.cfg.DATA_DIR, mask_rel)

            # Tracciamento file mancanti
            if not os.path.exists(orig_path): missing.append(orig_path)
            if not os.path.exists(mask_path): missing.append(mask_path)
            if not os.path.exists(manip_path): missing.append(manip_path)

            if os.path.exists(orig_path) and os.path.exists(mask_path) and os.path.exists(manip_path):
                # Usiamo il nome del file originale per ricavare lo stem
                stem = os.path.splitext(os.path.basename(orig_path))[0]
                
                rows.append({
                    "image_id": f"sagi_coco_{stem}", 
                    "orig_path": orig_path,
                    "manip_path": manip_path,
                    "mask_path": mask_path,
                    "category_name": str(r.get("type", "unknown")),
                    "target_prompt": str(r.get("prompt", "")),
                    "source": "sagi_hdpainter", 
                    "mask_type": "png",
                    "split": str(r.get("split", "train")) 
                })

        if missing:
            print(f"ATTENZIONE: {len(missing)} file mancanti (ne mostro 5):")
            for m in list(set(missing))[:5]:
                print(f"  - {m}")

        return pd.DataFrame(rows)

    def _undersample(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.cfg.DATASET_SIZE is None or self.cfg.DATASET_SIZE >= len(df):
            return df

        print(f"Esecuzione undersampling stratificato a {self.cfg.DATASET_SIZE} campioni...")
        
        split_counts = df["split"].value_counts(normalize=True)
        sampled_parts = []
        
        for split_name, proportion in split_counts.items():
            n_samples = int(self.cfg.DATASET_SIZE * proportion)
            split_group = df[df["split"] == split_name]
            
            n = min(n_samples, len(split_group))
            sampled_parts.append(split_group.sample(n=n, random_state=self.cfg.SEED))
            
        df_sampled = pd.concat(sampled_parts, ignore_index=True)
        
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

        manifest = self._undersample(manifest)
        print(f"Totale campioni finali: {len(manifest)}")

        # Riordino esplicito per forzare la struttura esatta dell'header richiesto
        expected_columns = [
            "image_id", "orig_path", "manip_path", "mask_path", 
            "category_name", "target_prompt", "source", "mask_type", "split"
        ]
        
        final_columns = [col for col in expected_columns if col in manifest.columns]
        manifest = manifest[final_columns]

        os.makedirs(os.path.dirname(self.cfg.MANIFEST_OUT), exist_ok=True)
        manifest.to_csv(self.cfg.MANIFEST_OUT, index=False)
        print(f"Manifest salvato con successo in: {self.cfg.MANIFEST_OUT}")

        return manifest