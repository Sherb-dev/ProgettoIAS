# pyright: reportOptionalSubscript=false

# ==============================================================================
# UTILS - Funzioni che principalmente plottano i grafici
# ==============================================================================
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from src.config import Config
from PIL import Image
from torch.utils.data import DataLoader
import torch


class Utils:

    # --------------------------------------------------------------------------
    # SALVATAGGIO METRICHE SU CSV
    # --------------------------------------------------------------------------

    @staticmethod
    def save_metrics_to_csv(
        results: dict,
        model_name: str,
        cfg: Config,
        threshold: float = 0.5,
    ) -> str:
        """
        Salva le metriche scalari di evaluate() in un CSV.
        Esclude fpr, tpr, labels, scores (non scalari).

        Args:
            results:     dizionario restituito da Evaluator.evaluate()
            model_name:  nome del modello (es. "AttentionUNet")
            cfg:         Config con FIGURES_DIR
            threshold:   soglia usata durante la valutazione

        Returns:
            path del CSV salvato
        """
        SCALAR_KEYS = ("mean_iou", "mean_f1", "mean_precision",
                       "mean_recall", "mean_bbox_iou", "mean_bbox_f1",
                       "mean_bbox_precision", "mean_bbox_recall",
                       "accuracy", "roc_auc")

        row = {"model": model_name, "threshold": threshold}
        row.update({k: results[k] for k in SCALAR_KEYS if k in results})

        df = pd.DataFrame([row])

        os.makedirs(cfg.FIGURES_DIR, exist_ok=True)
        out_path = os.path.join(cfg.FIGURES_DIR, f"{model_name}_metrics.csv")

        # Se il file esiste già, appende senza riscrivere l'header
        write_header = not os.path.exists(out_path)
        df.to_csv(out_path, mode="a", header=write_header, index=False)

        print(f"[Utils] Metriche salvate in: {out_path}")
        return out_path

    # --------------------------------------------------------------------------
    # VISUALIZZAZIONE METRICHE
    # --------------------------------------------------------------------------

    @staticmethod
    def plot_metrics(
        results: dict,
        model_name: str,
        cfg: Config,
        history: dict | None = None,
        save_file = False
    ) -> None:
        """
        Produce una figura con 3 pannelli:
          1. Bar chart delle metriche scalari (pixel-wise + bounding box + classificazione)
          2. Curva ROC
          3. Curva di loss train/val per epoca (opzionale, se history è fornita)

        Args:
            results:     dizionario restituito da Evaluator.evaluate()
            model_name:  nome del modello
            cfg:         Config con FIGURES_DIR
            history:     dizionario {"train_loss": [...], "val_loss": [...]}
                         restituito da Trainer.fit() — opzionale
        """
        has_history = history is not None and len(history.get("train_loss", [])) > 0
        n_cols = 3 if has_history else 2

        fig = plt.figure(figsize=(6 * n_cols, 5))
        gs  = gridspec.GridSpec(1, n_cols, figure=fig, wspace=0.35)

        # ------------------------------------------------------------------
        # Pannello 1 — Bar chart metriche scalari
        # ------------------------------------------------------------------
        ax1 = fig.add_subplot(gs[0, 0])

        SCALAR_KEYS   = ["mean_iou", "mean_f1", "mean_precision", "mean_recall",
                         "mean_bbox_iou", "mean_bbox_f1", "mean_bbox_precision", "mean_bbox_recall",
                         "accuracy", "roc_auc"]
        SCALAR_LABELS = ["IoU", "F1", "Precision", "Recall",
                         "BBox IoU", "BBox F1", "BBox Prec.", "BBox Rec.",
                         "Accuracy", "ROC AUC"]
        COLORS        = ["#4C72B0", "#55A868", "#C44E52", "#8172B2",
                         "#4C72B0", "#55A868", "#C44E52", "#8172B2",
                         "#CCB974", "#64B5CD"]

        values = [results.get(k, 0.0) for k in SCALAR_KEYS]

        bars = ax1.bar(SCALAR_LABELS, values, color=COLORS, edgecolor="white", width=0.6)
        # Le 4 barre "BBox *" riusano gli stessi colori delle 4 pixel-wise
        # corrispondenti (IoU/F1/Precision/Recall), ma con il bordo tratteggiato:
        # cosi' si vede a colpo d'occhio quali coppie di barre vanno confrontate.
        for bar in bars[4:8]:
            bar.set_hatch("//")

        ax1.set_ylim(0, 1.1)
        ax1.set_ylabel("Score")
        ax1.set_title(f"{model_name} — Metriche")
        ax1.tick_params(axis="x", rotation=45)

        # Valore sopra ogni barra
        for bar, val in zip(bars, values):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8
            )

        # ------------------------------------------------------------------
        # Pannello 2 — Curva ROC
        # ------------------------------------------------------------------
        ax2 = fig.add_subplot(gs[0, 1])

        fpr = results.get("fpr", [0, 1])
        tpr = results.get("tpr", [0, 1])
        auc_val = results.get("roc_auc", 0.0)

        ax2.plot(fpr, tpr, color="#4C72B0", lw=2,
                 label=f"AUC = {auc_val:.3f}")
        ax2.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax2.fill_between(fpr, tpr, alpha=0.08, color="#4C72B0")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title(f"{model_name} — ROC Curve")
        ax2.legend(loc="lower right")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1.05)

        # ------------------------------------------------------------------
        # Pannello 3 — Loss curve (opzionale)
        # ------------------------------------------------------------------
        if has_history:
            ax3 = fig.add_subplot(gs[0, 2])

            epochs = range(1, len(history["train_loss"]) + 1)
            ax3.plot(epochs, history["train_loss"], label="Train loss",
                     color="#C44E52", lw=2)
            ax3.plot(epochs, history["val_loss"],   label="Val loss",
                     color="#55A868", lw=2, linestyle="--")
            ax3.set_xlabel("Epoca")
            ax3.set_ylabel("Loss")
            ax3.set_title(f"{model_name} — Loss curve")
            ax3.legend()

        # ------------------------------------------------------------------
        # Salvataggio
        # ------------------------------------------------------------------
        if save_file:
            os.makedirs(cfg.FIGURES_DIR, exist_ok=True)
            out_path = os.path.join(cfg.FIGURES_DIR, f"{model_name}_dashboard.png")
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"[Utils] Figura salvata in: {out_path}")

        plt.show()

    """ model name inteso come il nome usato per i check point"""
    @staticmethod
    def plot_loss(model_name, cfg, save_file=False):
        history_csv_path = os.path.join(cfg.CHECKPOINT_DIR, f"{model_name}_history.csv")
        df = pd.read_csv(history_csv_path)

        plt.figure(figsize=(9, 5), dpi=120)
        plt.plot(df["epoch"], df["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
        plt.plot(df["epoch"], df["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--")

        plt.title(f"{model_name} Loss", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(frameon=True, fontsize=11)
        plt.tight_layout()

        if save_file:
            save_path = os.path.join(cfg.FIGURES_DIR, f"loss_plot_{model_name}.png")
            plt.savefig(save_path, dpi=300)
            print(f"Grafico salvato in: {save_path}")

        plt.show()

    #########################
    # UTILS ANALISI DATASET #
    #########################
    
    @staticmethod
    def analyze_image_sizes( #Estrae ed analizza la risoluzione (width x height) delle immagini pre e post inpainting
    dataset_name: str,
    csv_path: Path,
    base_dir: Path,
    col_orig,
    col_inpainted,
    subfolder_orig: str = "0",
    subfolder_inpainted: str = "1",
    ): 

        df = pd.read_csv(csv_path) #legge il csv da csv_path e lo carica in un DataFrame
        records = [] #lista vuota per raccogliere i dizionari con i dati delle dimensioni

        for _, row in df.iterrows(): #scorre ogni riga del dataframe
            #cicla sulle due fasi (pre e post inpainting), associa sottocartella e colonna rispettive
            for stage, subfolder, col in ( 
                ("pre_inpainting", subfolder_orig, col_orig),
                ("post_inpainting", subfolder_inpainted, col_inpainted),
            ):
                # risoluzione dinamica del filename: se 'col' è una funzione (callback) la esegue sulla riga 
                # altrimenti legge direttamente il valore del campo                
                filename = col(row) if callable(col) else row[col]
                
                # compone il percorso completo del file
                fpath = base_dir / subfolder / filename 

                if fpath.is_file(): #verifica se il file esiste
                    try:
                        with Image.open(fpath) as im: #apre l'immagine con Pillow
                            w, h = im.size #estrae larghezza e altezza (w e h) in pixel
                        records.append( #aggiunge i dati estratti alla loista dei record come dizionario
                            {"stage": stage, "width": w, "height": h}
                        )
                    except Exception:
                        continue #se l'immagine è corrotta/non si apre passa alla successiva

        if not records: #controlla se non è stato estratto alcun record valido
            print(
                f"[{dataset_name}] Nessuna dimensione trovata: controllare i percorsi."
            )
            return None

        #calcolo metriche
        size_df = pd.DataFrame(records) #nuovo DataFrame
        #calcolo della colonna 'megapixels' come larghezza * altezza / 1.000.000
        size_df["megapixels"] = size_df["width"] * size_df["height"] / 1e6
        colors = {"pre_inpainting": "steelblue", "post_inpainting": "darkorange"} #colori da utilizzare per le diverse fasi

        # --- Grafico 1: Pre-inpainting ---
        df_pre = size_df[size_df["stage"] == "pre_inpainting"] #tiene solo i dati relativi al pre inpainting
        if not df_pre.empty: #se ci sono dati disponibili procede alla generazione di grafici
            fig1, axes1 = plt.subplots(1, 3, figsize=(16, 4.5)) #3 subplot affiancati
            
            #primo subplot: istogramma delle larghezze in pixel
            axes1[0].hist( # 30 intervalli, steelblue, 70% opacità
                df_pre["width"], 
                bins=30, 
                color=colors["pre_inpainting"], 
                alpha=0.7 
            )
            #imposta titolo e etichetta dell'asse X per il subplot
            axes1[0].set_title(f"{dataset_name} (Pre-Inpainting) - Larghezza") 
            axes1[0].set_xlabel("width (px)")

            #secondo subplot: istogramma delle altezze in pixel
            axes1[1].hist(
                df_pre["height"], 
                bins=30, 
                color=colors["pre_inpainting"], 
                alpha=0.7
            )
            #imposta titolo e etichetta dell'asse X per il subplot
            axes1[1].set_title(f"{dataset_name} (Pre-Inpainting) - Altezza")
            axes1[1].set_xlabel("height (px)")

            #terzo subplot: scatterplot larghezza vs altezza
            axes1[2].scatter(
                df_pre["width"],
                df_pre["height"],
                color=colors["pre_inpainting"],
                alpha=0.5, #opacità 50%
            )
            #imposta titolo e etichette degli assi per il subplot
            axes1[2].set_title(f"{dataset_name} (Pre-Inpainting) - Width vs Height")
            axes1[2].set_xlabel("width (px)")
            axes1[2].set_ylabel("height (px)")

            plt.tight_layout() #regolazione automatica della spaziatura
            plt.show()

        # --- Grafico 2: Post-inpainting ---
        df_post = size_df[size_df["stage"] == "post_inpainting"] #tiene solo i dati relativi al post inpainting
        if not df_post.empty: #se ci sono dati disponibili procede alla generazione di grafici
            fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4.5)) #crea una sceonda figura con 3 subplot

            #primo subplot: istogramma delle larghezze in pixel
            axes2[0].hist( # 30 intervalli, darkorange, 70% opacità
                df_post["width"],
                bins=30,
                color=colors["post_inpainting"], 
                alpha=0.7,
            )
            #imposta titolo e etichetta dell'asse X per il subplot
            axes2[0].set_title(f"{dataset_name} (Post-Inpainting) - Larghezza")
            axes2[0].set_xlabel("width (px)")

            #secondo subplot: istogramma delle altezze in pixel
            axes2[1].hist(
                df_post["height"],
                bins=30,
                color=colors["post_inpainting"],
                alpha=0.7,
            )
            #imposta titolo e etichetta dell'asse X per il subplot
            axes2[1].set_title(f"{dataset_name} (Post-Inpainting) - Altezza")
            axes2[1].set_xlabel("height (px)")

            #terzo subplot: scatterplot larghezza vs altezza
            axes2[2].scatter(
                df_post["width"],
                df_post["height"],
                color=colors["post_inpainting"],
                alpha=0.5,
            )
            #imposta titolo e etichette degli assi X e Y per il subplot
            axes2[2].set_title(f"{dataset_name} (Post-Inpainting) - Width vs Height")
            axes2[2].set_xlabel("width (px)")
            axes2[2].set_ylabel("height (px)")

            plt.tight_layout() #regolazione automatica della spaziatura
            plt.show()

        #print(size_df.groupby("stage")[["width", "height", "megapixels"]].describe()) #statistiche descrittive raggruppate per fase
        return size_df


    @staticmethod
    def analyze_image_sizes_combined(
    size_df_a: pd.DataFrame,
    name_a: str,
    size_df_b: pd.DataFrame,
    name_b: str,
    ):

        #controlla se uno dei due DataFrame passati in ingresso è None
        if size_df_a is None or size_df_b is None:
            print("Impossibile creare il grafico combinato: uno dei due dataset è vuoto.")
            return None

        #unisce i due DataFrame lungo le righe per ottenerne uno unico
        combined = pd.concat([size_df_a, size_df_b], ignore_index=True) 

        for stage, stage_label in ( #cicla sulle due fasi definendo il nome interno e l'etichetta
            ("pre_inpainting", "Pre-inpainting"),
            ("post_inpainting", "Post-inpainting"),
        ):
            stage_df = combined[combined["stage"] == stage] #mantiene i dati solo per la fase corrente
            if stage_df.empty: #verifica se il DataFrame è vuoto
                print(f"Nessun dato per lo stage '{stage}', grafico saltato.")
                continue

            #inizializza una figura con tre subplot
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

            #primo subplot: istogramma normalizzato della larghezza in pixel
            axes[0].hist(stage_df["width"], bins=30, density=True, color="steelblue")
            #secondo subplot: istogramma normalizzato dell'altezza in pixel
            axes[1].hist(stage_df["height"], bins=30, density=True, color="steelblue")
            #terzo subplot: scatterplot larghezza vs altezza
            axes[2].scatter(stage_df["width"], stage_df["height"], alpha=0.4, color="steelblue")

            #imposta l'etichetta dell'asse Y per i primi due subplot
            axes[0].set_ylabel("densità")
            axes[1].set_ylabel("densità")

            #imposta titolo ed etichetta dell'asse X per il primo subplot
            axes[0].set_title(f"Larghezza - {stage_label} ({name_a} + {name_b})")
            axes[0].set_xlabel("width (px)")

            #imposta titolo ed etichetta dell'asse X per il secondo subplot
            axes[1].set_title(f"Altezza - {stage_label} ({name_a} + {name_b})")
            axes[1].set_xlabel("height (px)")

            #imposta titolo ed etichette degli assi per il terzo subplot
            axes[2].set_title(f"Width vs Height - {stage_label} ({name_a} + {name_b})")
            axes[2].set_xlabel("width (px)")
            axes[2].set_ylabel("height (px)")

            plt.tight_layout() #regolazione automatica del layout
            plt.show()

            #stampa numerosità e statistiche descrittive
            #print(f"--- {stage_label} (n={len(stage_df)}) ---")
            #print(stage_df[["width", "height", "megapixels"]].describe())

        #return combined #restituisce il dataframe combinato


    # Utilizzato da analyze_mask_sizes, ho preferito non fare una funzione interna per migliorare la leggibilità
    # Carica e normalizza una maschera di inpainting da disco come array 2D NumPy.
    # Gestisce sia i file matriciali .npy di Bagel sia i file immagine standard come .png o .jpg di COCO,
    # convertendoli in una matrice 2D scala di grigi [0, 255].
    @staticmethod
    def load_mask(fpath: Path) -> np.ndarray:
        ext = fpath.suffix.lower() # estrae l'estensione del file
        # caso Bagel, estensione .npy
        if ext == ".npy":
            arr = np.load(fpath) #carica il file in un array
            arr = np.squeeze(arr)  # rimuove eventuali dimensioni singole
            if arr.ndim != 2: #verifica che l'array sia bidimensionale
                raise ValueError(f"Maschera .npy con shape inattesa: {arr.shape} ({fpath})")
            # normalizza a range 0-255
            if arr.dtype == bool: #se l'array è di bool
                arr = arr.astype(np.uint8) * 255 #true = 255, false = 0
            elif arr.max() <= 1.0: #se i valori sono in virgola mobile li scala in [0, 255]
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8) #negli altri casi conversione a uinit8
            return arr
        #caso COCO, estensione png
        else:
            with Image.open(fpath) as im: #apre l'immagine dal percorso
                return np.array(im.convert("L")) #converte l'immagine in scala di grigi e la trasforma direttamente in array


    # analizza le maschere calcolando sia l'estensione assoulta della regione modificata sia l'impatto relativo all'immagine di partenza.  
    # Si utilizzano due callback che, data una riga del CSV, 
    # restituiscono rispettivamente il nome del file maschera e il nome del file immagine originale (cartella '0')
    # da usare come riferimento per calcolare la percentuale di pixel modificati.
    @staticmethod
    def analyze_mask_sizes(
        dataset_name: str,
        csv_path: Path,
        base_dir: Path,
        mask_name_fn,
        image_name_fn,
        image_subfolder: str = "0",
        mask_subfolder: str = "masks",
        threshold: int = 127,
    ):

        #costruisce i percorsi delle cartelle per maschere e imamgini
        mask_dir = base_dir / mask_subfolder
        image_dir = base_dir / image_subfolder
        if not mask_dir.is_dir(): #verifica che la directory esista
            print(f"[{dataset_name}] Cartella maschere non trovata: {mask_dir}")
            return None

        df = pd.read_csv(csv_path) #legge il csv e crea il dataframe
        records = [] #inizializza la lista per salvare le metriche e il contatore di errori
        n_errors = 0

        for _, row in df.iterrows(): #scorre ogni riga del DataFrame
            #risolve i nomi dei file di maschera e immagine
            mask_fname = mask_name_fn(row)
            image_fname = image_name_fn(row)
            #compone i percorsi
            mask_path = mask_dir / mask_fname
            image_path = image_dir / image_fname

            try:
                #caricamento maschera come array
                arr = Utils.load_mask(mask_path)
                h, w = arr.shape #estrae altezza e larghezza 
                canvas_px = w * h #calcola l'area

                #calcolo pixel effettivamente manipolati (con soglia superiore a quella definita)
                manipulated_px = int(np.count_nonzero(arr > threshold))

                with Image.open(image_path) as im: #apre l'immagine corrispondente per estrarne le dimensioni
                    img_w, img_h = im.size
                img_area = img_w * img_h #calcola l'area totale

                #calcolo della percentuale di area manipolata rispetto all'immagine
                pct_manipulated = (manipulated_px / img_area) * 100 if img_area > 0 else np.nan

                records.append( #aggiunge tutte le metriche ottenute com dizioanrio alla lista dei record
                    {
                        "mask_filename": mask_fname,
                        "width": w,
                        "height": h,
                        "canvas_px": canvas_px,
                        "area_px": manipulated_px,
                        "image_width": img_w,
                        "image_height": img_h,
                        "pct_manipulated": pct_manipulated,
                    }
                )
            except Exception:
                n_errors += 1
                continue

        if not records: #verifica se è stata letta almeno una maschera
            print(f"[{dataset_name}] Nessuna maschera leggibile in {mask_dir}")
            return None

        mask_df = pd.DataFrame(records) #converte records in un DataFrame

        #crea una figura con due subplot
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        #primo subplot: area assoluta
        axes[0].hist(mask_df["area_px"], bins=30, color="seagreen")
        axes[0].set_title(f"{dataset_name} - Area maschere (px)")
        axes[0].set_xlabel("area (px^2)")

        #secondo subplot: area relativa
        axes[1].hist(mask_df["pct_manipulated"], bins=30, color="firebrick")
        axes[1].set_title(f"{dataset_name} - % pixel immagine manipolati")
        axes[1].set_xlabel("% pixel manipolati")

        plt.tight_layout() #regolazione automatica degli spazi
        plt.show()
        print(f"[{dataset_name}] Grafico dimensioni maschere mostrato (righe con errori/mancanti: {n_errors}).")
        #print(mask_df[["width", "height", "area_px", "pct_manipulated"]].describe()) #mostra il sommario della statistiche
        return mask_df
    
    #unisce le informazioni riguardanti le maschere
    @staticmethod
    def analyze_mask_sizes_combined(
        mask_df_a: pd.DataFrame | None,
        mask_df_b: pd.DataFrame | None,
    ):

        if mask_df_a is None or mask_df_b is None: #verifica se uno o entrambi i DataFrame in ingresso sono none
            print("Impossibile creare il grafico combinato: uno dei due dataset è vuoto.")
            return None

        #unisce i due DataFrame lungo le righe
        combined = pd.concat([mask_df_a, mask_df_b], ignore_index=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5)) #inizializza una figura con due subplot
        #primo subplot: istogramma dell'area assoluta
        axes[0].hist(combined["area_px"], bins=30, color="seagreen")
        axes[0].set_title("Area maschere (px) - dataset combinato")
        axes[0].set_xlabel("area (px^2)")

        #secondo subplot: istogramma dell'area relativa
        axes[1].hist(combined["pct_manipulated"], bins=30, color="firebrick")
        axes[1].set_title("% pixel immagine manipolati - dataset combinato")
        axes[1].set_xlabel("% pixel manipolati")

        plt.tight_layout() #regolazione automatica della spaziatura
        plt.show()

        print(f"Grafico dimensioni maschere combinato mostrato (n={len(combined)}).")
        #print(combined[["area_px", "pct_manipulated"]].describe()) #mostra le statistiche descrittive
        #return combined

    # La funzione analyze_category_distribution() analizza e visualizza la frequenza delle categorie all'interno del dataset.  
    # Legge il file CSV, calcola le occorrenze per ciascuna categoria e genera un grafico a barre per le prime `top_n`classi più frequenti.
    
    @staticmethod
    def analyze_category_distribution(
        dataset_name: str,
        csv_path: Path,
        category_col: str,
        top_n: int = 20,
    ):

        df = pd.read_csv(csv_path) #legge il csv specificato e crea un DataFrame
        counts = df[category_col].value_counts()#calcolo delle frequenze per ogni categoria

        top_counts = counts.head(top_n) #selezione top n categorie

        #inizializza la figura con dimensione verticale dinamica
        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * min(top_n, len(counts)))))
        #crea un grafico a barre orizzontali
        ax.barh(top_counts.index[::-1], top_counts.values[::-1], color="steelblue")
        #impostazione etichette e titolo
        ax.set_xlabel("Numero di immagini")
        ax.set_ylabel(category_col)
        ax.set_title(f"{dataset_name} - Distribuzione categorie (top {top_n} di {len(counts)})")

        plt.tight_layout() # reagolazione automatica della spaziatura
        plt.show()
        #print(counts)
        return counts

    #genera il grafico combinato
    @staticmethod
    def analyze_category_distribution_combined(
        counts_a: pd.Series | None,
        counts_b: pd.Series | None,
        top_n: int = 20,
    ):

        if counts_a is None or counts_b is None: # verifica se una delle due serie è none
            print("Impossibile creare il grafico combinato: uno dei due dataset è vuoto.")
            return None

        #somma elemento per elemento i conteggi dei due dataset
        combined = counts_a.add(counts_b, fill_value=0).astype(int).sort_values(ascending=False)

        #inizializza la figura con altezza dinamica
        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * min(top_n, len(combined)))))
        top_counts = combined.head(top_n) #seleziona le prime top n categorie
        #crea un grafico a barre orizzontali
        ax.barh(top_counts.index[::-1], top_counts.values[::-1], color="steelblue")
        ax.set_xlabel("Numero di immagini")
        ax.set_ylabel("categoria")
        ax.set_title(f"Distribuzione categorie combinata (top {top_n} di {len(combined)})")

        plt.tight_layout() #regoalzione automatica della spaziatura
        plt.show()

        #print(f"Grafico distribuzione categorie combinato mostrato (totale righe: {combined.sum()}).")
        #print(combined)
        #return combined


    # Legge tutte le maschere dalla directory e le ridimensione alla dimensione target,
    # le binarizza e le accumula in una matrice di conteggio per la visualizzazione della densità spaziale.
    @staticmethod
    def accumulate_mask_heatmap(
    base_dir: Path,
    mask_subfolder: str,
    target_size: tuple[int, int],
    threshold: int,
    ):

        mask_dir = base_dir / mask_subfolder #compone il percorso completo
        if not mask_dir.is_dir(): #controlla che il percorso esista
            return None, 0

        #inizializazione accumulatore
        accumulator = np.zeros(target_size[::-1], dtype=np.float64)  # inverte le dimensioni (H, W)
        n_masks = 0 #contatore numero totale maschere

        #iterazione e accumulo
        for fname in os.listdir(mask_dir): #scorre i nomi dei file all'interno della cartella
            fpath = mask_dir / fname #compone il percorso completo
            try:
                arr = Utils.load_mask(fpath) #maschera come array npy
                #ridimensionamento alla dimensione target
                im_resized = np.array(
                    Image.fromarray(arr).resize(target_size, Image.NEAREST)
                )
                #binarizzazione
                binary = (im_resized > threshold).astype(np.float64)
                accumulator += binary #somma la maschera binarizzata alla amtrice accumulatore
                n_masks += 1 #aumento contatore maschere
            except Exception:
                continue

        return accumulator, n_masks #restituisce matrice accumulatore e numero maschere


    # Costruisce una heatmap aggregata che mostra in quali zone sono mediamente più presenti le maschere.
    # Ogni maschera viene ridimensionata, binarizzata e sommata; il risultato è normalizzato in [0, 1].
    @staticmethod
    def analyze_mask_heatmap(
        dataset_name: str,
        base_dir: Path,
        mask_subfolder: str = "masks",
        target_size: tuple[int, int] = (256, 256),
        threshold: int = 127,
    ):
        #chiama la fuznione helper per sommare le maschere 
        accumulator, n_masks = Utils.accumulate_mask_heatmap(base_dir, mask_subfolder, target_size, threshold)

        if accumulator is None: #se la matrice è none
            print(f"[{dataset_name}] Cartella maschere non trovata: {base_dir / mask_subfolder}")
            return None
        if n_masks == 0: #se non sono state elaborate maschere
            print(f"[{dataset_name}] Nessuna maschera leggibile per la heatmap.")
            return None

        #calcola la heatmap 
        heatmap = accumulator / n_masks  

        fig, ax = plt.subplots(figsize=(6, 5)) #inizializza figura
        im = ax.imshow(heatmap, cmap="inferno")#visualizza la heatmap con mappa di colori inferno
        ax.set_title(f"{dataset_name} - Heatmap posizione maschere (n={n_masks})")
        ax.set_xticks([]) #nasconde tick asse x
        ax.set_yticks([]) #nasconde tick asse y
        fig.colorbar(im, ax=ax, label="Frequenza presenza maschera") #barra laterale per la scala di frequenza

        plt.tight_layout() #regolazione spaziatura automatica
        plt.show()
        #return heatmap


    #heatmap combinata
    @staticmethod
    def analyze_mask_heatmap_combined(
        base_dir_a: Path,
        base_dir_b: Path,
        mask_subfolder: str = "masks",
        target_size: tuple[int, int] = (256, 256),
        threshold: int = 127,
    ):

        #chiama la funzione helper per accumulare le maschere dei due dataset
        acc_a, n_a = Utils.accumulate_mask_heatmap(base_dir_a, mask_subfolder, target_size, threshold)
        acc_b, n_b = Utils.accumulate_mask_heatmap(base_dir_b, mask_subfolder, target_size, threshold)

        if acc_a is None or acc_b is None:
            print("Impossibile creare la heatmap combinata: una delle due cartelle non esiste.")
            return None
        if n_a + n_b == 0:
            print("Nessuna maschera leggibile in nessuno dei due dataset.")
            return None

        heatmap = (acc_a + acc_b) / (n_a + n_b) #calcola la heatmap globale 

        fig, ax = plt.subplots(figsize=(6, 5)) #inizializza il plot
        im = ax.imshow(heatmap, cmap="inferno") #visualizza heatmap 
        ax.set_title(f"Heatmap posizione maschere combinata (n={n_a + n_b})")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, label="Frequenza presenza maschera")

        plt.tight_layout()
        plt.show()
        print(f"Heatmap maschere combinata mostrata (n={n_a + n_b}, COCO={n_a}, Bagel={n_b}).")
        #return heatmap

        

    ###################################
    #   ALTRE FUNZIONI DA COMMENTARE  #
    ###################################

    # Definisce la funzione per costruire il percorso completo dell'immagine inpainted a partire dal suo ID
    @staticmethod
    def manipulated_path_for(image_id) -> str:
        """Le immagini manipolate sul disco sono nominate come {image_id}.jpg."""
        return os.path.join(Config.MANIP_DIR, f"{image_id}.jpg")
    @staticmethod
    def analisi_immagini_nere(df):
        """
        Controlla, per ogni immagine manipolata, se risulta completamente nera
        (tutti i pixel a 0) o illeggibile/corrotta: sono scarti dovuti a falsi
        positivi del filtro NSFW di PowerPaint, non utili per il training.
        Mostra un grafico a torta con il rapporto nere/utilizzabili.
        """
        #inizializzazione contatori e lista
        n_nere, n_utili = 0, 0
        esempi_nere = []

        for _, row in df.iterrows():#per ciascuna riga
            p = Utils.manipulated_path_for(row["image_id"])#trova il percorso
            try:
                with Image.open(p) as img:
                    arr = np.array(img) #converte l'immagine in .npy
                if not np.any(arr): #se è tutta nera
                    n_nere += 1
                    esempi_nere.append(row["image_id"])
                else: #se ha almeno un pixel diverso da 0
                    n_utili += 1
            except Exception as e:
                print(f"Errore lettura {p}: {e}")
                n_nere += 1
                esempi_nere.append(row["image_id"])

        totale = n_nere + n_utili
        print(f"Nere/corrotte: {n_nere}  ({n_nere / totale * 100:.2f}%)")
        print(f"Utili per il training: {n_utili}  ({n_utili / totale * 100:.2f}%)")
        if esempi_nere[:10]:
            print(f"Esempi di image_id nere: {esempi_nere[:10]}")

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(
            [n_utili, n_nere],
            labels=[f"Utili ({n_utili})", f"Nere/scarti ({n_nere})"],
            autopct="%1.1f%%",
            colors=["steelblue", "indianred"],
            startangle=90,
        )
        ax.set_title("Rapporto immagini utili vs completamente nere")
        plt.tight_layout()
        plt.show()

        return {"nere": n_nere, "utili": n_utili}

    @staticmethod
    def plot_undersampling(coco_pre_us, bagel_pre_us, manifest_post_us):
        """
        Confronta il numero di campioni di MS-COCO e Bagel prima e dopo
        l'undersampling per visualizzare lo squilibrio originale
        (10k vs 4k) e il riequilibrio ottenuto.
        """

        conteggio_dopo = manifest_post_us["source"].value_counts() #frequenza di ogni source

        etichette = ["MS-COCO", "Bagel"]
        valori_prima = [len(coco_pre_us), len(bagel_pre_us)] #elementi pre undersampling
        valori_dopo = [conteggio_dopo.get("ds1", 0), conteggio_dopo.get("ds2", 0)] #conteggio estratto dal dataset

        x = range(len(etichette))
        larghezza = 0.35

        fig, ax = plt.subplots(figsize=(6, 4.5))
        barre_prima = ax.bar([i - larghezza/2 for i in x], valori_prima,
                            larghezza, label="Prima", color="lightsteelblue")
        barre_dopo = ax.bar([i + larghezza/2 for i in x], valori_dopo,
                            larghezza, label="Dopo", color="steelblue")

        ax.set_xticks(list(x))
        ax.set_xticklabels(etichette)
        ax.set_ylabel("Numero di campioni")
        ax.set_title("Effetto dell'undersampling per sorgente")
        ax.legend()

        for barre in (barre_prima, barre_dopo):
            for b in barre:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width()/2, h + 50, str(int(h)),
                        ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_split(manifest):
        """
        Mostra, per ogni split (train/val/test), quanti campioni appartengono
        a ds1 (MS-COCO) e a ds2 (Bagel). Verifica visivamente che lo split
        stratificato abbia mantenuto la proporzione tra le due sorgenti.
        """
        #raggruppa per split e source, calcola frequenza e riorganizza la tabella
        tabella = manifest.groupby(["split", "source"]).size().unstack(fill_value=0)

        ordine_split = [s for s in ["train", "val", "test"] if s in tabella.index] #definisce l'ordine logico
        tabella = tabella.reindex(ordine_split) #riordina le righe

        nomi = {"ds1": "MS-COCO", "ds2": "Bagel"}
        tabella = tabella.rename(columns=nomi)

        print(tabella)

        ax = tabella.plot(kind="bar", figsize=(7, 5), color=["steelblue", "indianred"])
        ax.set_xlabel("Split")
        ax.set_ylabel("Numero di campioni")
        ax.set_title("Distribuzione dei campioni per split e per sorgente")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Sorgente")

        for container in ax.containers:
            ax.bar_label(container, fontsize=9)

        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_esempi_augmentation(manifest_df, transform, n_esempi=10, seed=42):#TODO cambiare il seed
        """
        Sceglie n_esempi campioni casuali dal manifest e mostra, per ciascuno,
        l'originale (manipolata) affiancata alla versione con data augmentation
        applicata.
        """

        sample = manifest_df.sample(n=min(n_esempi, len(manifest_df)), random_state=seed) #selezione casuale del sample

        fig, axes = plt.subplots(len(sample), 2, figsize=(6, 3 * len(sample)))

        for i, (_, row) in enumerate(sample.iterrows()): #itera su ogni elemento
            img = Image.open(row["manip_path"]).convert("RGB")
            mask_vuota = Image.new("L", img.size, 0)  # non ci serve la maschera per questo task

            img_aug, _ = transform(img, mask_vuota)
            #cambia l'ordine dei canali, lo converte in .npy e lo denoramlizza
            img_aug_np = (img_aug.permute(1, 2, 0).numpy() * 0.5) + 0.5

            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"Originale (id={row['image_id']})", fontsize=9)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(img_aug_np.clip(0, 1))
            axes[i, 1].set_title("Augmentata", fontsize=9)
            axes[i, 1].axis("off")

        plt.tight_layout()
        plt.show()


    # Definisce la funzione per confrontare n immagini originali  con le rispettive versioni elaborate e normalizzate nel PyTorch Dataset
    @staticmethod
    def plot_confronto_normalizzazione(dataset, n_immagini=1, seed=Config.SEED):
        """
        Confronta n_immagini originali (lette da disco, valori [0,255]) con le
        stesse immagini dentro un batch di training,
        cioè già passate per JointTransform (resize + normalizzazione in [-1,1],
        ed eventuale augmentation se il dataset ha train=True).
        """
        # inizializza un generatore casuale NumPy con il seed specificato per garantire la riproducibilità
        rng = np.random.default_rng(seed)

        # recupera il numero totale di righe disponibili nel DataFrame associato al dataset
        n_righe = len(dataset.rows)

        # simita n_immagini al numero totale di righe presenti nel dataset per evitare errori
        n_immagini = min(n_immagini, n_righe)

        # seleziona n_immagini indici di riga casuali senza reinserimento
        righe_idx = rng.choice(n_righe, size=n_immagini, replace=False)

        fig, axes = plt.subplots(2 * n_immagini, 2, figsize=(11, 4.5 * n_immagini))

        # assicura che axes sia un array 2D anche quando n_immagini = 1
        if n_immagini == 1:
            axes = axes.reshape(2, 2)

        # itera su ciascuna immagine da analizzare
        for i, riga_idx in enumerate(righe_idx):
            riga_img = i * 2
            riga_hist = riga_img + 1

            # calcola l'indice del dataset corrispondente (forzando un numero dispari per selezionare la variante "manipolata")
            idx_dataset = riga_idx * 2 + 1  # forza il caso "manipolata"

            # estrae la riga dal DataFrame associata all'indice casuale generato
            row = dataset.rows.iloc[riga_idx]
            img_originale = Image.open(row["manip_path"]).convert("RGB")
            # converte l'immagine PIL in un array NumPy con valori interi nell'intervallo [0, 255]
            arr_originale = np.array(img_originale)

            image_tensor, _, _ = dataset[idx_dataset]
            # trasforma le dimensioni del tensore da (C, H, W) a (H, W, C) e lo converte in array NumPy (valori attesi in [-1, 1])
            arr_normalizzato = image_tensor.permute(1, 2, 0).numpy()

            axes[riga_img, 0].imshow(arr_originale)
            axes[riga_img, 0].set_title(f"Originale (id={row['image_id']})\nrange [{arr_originale.min()}, {arr_originale.max()}]")
            axes[riga_img, 0].axis("off")

            axes[riga_img, 1].imshow(np.clip(arr_normalizzato, 0, 1))
            axes[riga_img, 1].set_title(f"Dentro il batch\n"
                                        f"range reale nel tensore: [{arr_normalizzato.min():.2f}, {arr_normalizzato.max():.2f}]")

            axes[riga_img, 1].axis("off")
            axes[riga_hist, 0].hist(arr_originale.flatten(), bins=50, color="steelblue")
            axes[riga_hist, 0].set_title("Istogramma - valori grezzi [0, 255]")
            axes[riga_hist, 0].set_xlabel("Valore pixel")

            axes[riga_hist, 1].hist(arr_normalizzato.flatten(), bins=50, color="steelblue")
            axes[riga_hist, 1].set_title("Istogramma - valori nel batch [-1, 1]")
            axes[riga_hist, 1].set_xlabel("Valore pixel")

        plt.suptitle("Confronto: immagini originali vs stesse immagini nel batch normalizzato", y=1.002)
        plt.tight_layout()
        plt.show()


    # definisce la funzione per visualizzare il triplo confronto (Originale grezza da disco, Maschera, Inpainted dal dataset)
    @staticmethod
    def plot_esempi_inpainting(dataset, n_esempi=3, seed=42): #TODO cambiare seed
        """
        Mostra un confronto affiancato per un numero variabile di esempi (n_esempi):
        - Immagine originale grezza (letta direttamente da disco, senza data augmentation)
        - Maschera di inpainting (dal dataset)
        - Immagine finale (manipolata/inpainted, dal dataset con trasformazioni)
        """
        rng = np.random.default_rng(seed)

        # recupera il numero totale di righe disponibili nel DataFrame associato al dataset
        n_righe = len(dataset.rows)

        # limita n_esempi al numero massimo di righe presenti nel dataset per evitare errori di indice
        n_esempi = min(n_esempi, n_righe)

        righe_idx = rng.choice(n_righe, size=n_esempi, replace=False) # seleziona casualmente n_esempi indici dal dataset senza reinserimento

        fig, axes = plt.subplots(n_esempi, 3, figsize=(12, 4 * n_esempi))

        # assicura che axes sia un array 2D anche se si richiede un solo esempio (n_esempi = 1)
        if n_esempi == 1:
            axes = np.expand_dims(axes, axis=0)

        # itera su ciascun esempio da mostrare
        for i, riga_idx in enumerate(righe_idx):
            # estrae la riga dal DataFrame per ricavare i percorsi originali dei file e le meta-informazioni
            row = dataset.rows.iloc[riga_idx]
            image_id = row["image_id"]

            img_orig = Image.open(row["orig_path"]).convert("RGB")

            idx_manip = riga_idx * 2 + 1 # calcola l'indice dispari per caricare la versione manipolata
            img_manip_tensor, mask_tensor, _ = dataset[idx_manip]

            # converte il tensore PyTorch dell'immagine manipolata (C, H, W) in array .npy (H, W, C) e denormalizza
            img_manip_np = (img_manip_tensor.permute(1, 2, 0).numpy() * 0.5) + 0.5

            # gestisce il tensore della maschera (se a canale singolo C=1 o bidimensionale)
            if mask_tensor.ndim == 3:
                mask_np = mask_tensor.squeeze(0).numpy()
            else:
                mask_np = mask_tensor.numpy()

            axes[i, 0].imshow(img_orig)
            axes[i, 0].set_title(f"Originale (id={image_id})", fontsize=10)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(mask_np, cmap="gray")
            axes[i, 1].set_title("Maschera", fontsize=10)
            axes[i, 1].axis("off")

            axes[i, 2].imshow(np.clip(img_manip_np, 0, 1))
            axes[i, 2].set_title("Inpainted", fontsize=10)
            axes[i, 2].axis("off")

        plt.suptitle("Confronto Inpainting: Originale vs Maschera vs Inpainted", y=1.002, fontsize=12)
        plt.show()



# Visualizza alcune predizioni sul test set
class AnalisiPredizioni:
    @staticmethod
    def compare_and_debug_errors(loaded_models, test_loader, cfg, n_errors=12, save_fig=False):
        # Estrae i due modelli e i loro nomi dalla lista passata
        model1, name1 = loaded_models[0]
        model2, name2 = loaded_models[1]
        
        error_cases = []
        
        with torch.no_grad():
            for images, masks, labels in test_loader:
                for i in range(images.size(0)):
                    img   = images[i]
                    gt    = masks[i]
                    label = int(labels[i].item())
                    
                    # Inferenza Modello 1
                    seg_logits1, cls_logits1 = model1(img.unsqueeze(0).to(cfg.DEVICE))
                    prob_mask1 = torch.sigmoid(seg_logits1).squeeze().cpu()
                    cls_score1 = torch.sigmoid(cls_logits1).item()
                    pred_label1 = int(cls_score1 >= cfg.CLS_THRESHOLD) # Usa la soglia config
                    
                    # Inferenza Modello 2
                    seg_logits2, cls_logits2 = model2(img.unsqueeze(0).to(cfg.DEVICE))
                    prob_mask2 = torch.sigmoid(seg_logits2).squeeze().cpu()
                    cls_score2 = torch.sigmoid(cls_logits2).item()
                    pred_label2 = int(cls_score2 >= cfg.CLS_THRESHOLD) # Usa la soglia config
                    
                    # se modello 1 sbaglia OPPURE modello 2 sbaglia
                    if (pred_label1 != label) or (pred_label2 != label):
                        error_cases.append({
                            "img": img,
                            "gt_mask": gt,
                            "label": label,
                            "mask1": prob_mask1 > cfg.SEG_THRESHOLD, # Usa la soglia segmentazione config
                            "pred1": pred_label1,
                            "score1": cls_score1,
                            "mask2": prob_mask2 > cfg.SEG_THRESHOLD, # Usa la soglia segmentazione config
                            "pred2": pred_label2,
                            "score2": cls_score2
                        })
                    
                    if len(error_cases) == n_errors:
                        break
                if len(error_cases) == n_errors:
                    break
        
        # Genera la galleria 12x4
        fig, axes = plt.subplots(n_errors, 4, figsize=(16, 4 * n_errors))
        
        for i, case in enumerate(error_cases):
            img_np = (case["img"].permute(1,2,0).cpu().numpy() * 0.5 + 0.5).clip(0,1)
            
            axes[i][0].imshow(img_np)
            axes[i][0].set_title(f"Input | Label Vera: {case['label']}")
            axes[i][0].axis('off')
            
            axes[i][1].imshow(case["gt_mask"].squeeze().cpu(), cmap="gray")
            axes[i][1].set_title("GT Mask")
            axes[i][1].axis('off')
            
            axes[i][2].imshow(case["mask1"], cmap="gray")
            color1 = "red" if case["pred1"] != case["label"] else "green"
            axes[i][2].set_title(f"{name1}\nPred: {case['pred1']} (p={case['score1']:.2f})", color=color1)
            axes[i][2].axis('off')
            
            axes[i][3].imshow(case["mask2"], cmap="gray")
            color2 = "red" if case["pred2"] != case["label"] else "green"
            axes[i][3].set_title(f"{name2}\nPred: {case['pred2']} (p={case['score2']:.2f})", color=color2)
            axes[i][3].axis('off')
            
        plt.tight_layout()
        if save_fig:
            plt.savefig(os.path.join(cfg.FIGURES_DIR, "galleria_errori.png"))
        plt.show()


    # Controlla la distribuzione degli score
    @staticmethod
    def plot_score_distribution(results, cfg, save_plot=False):
        scores = np.array(results["scores"])
        labels = np.array(results["labels"])
        
        plt.figure(figsize=(8, 4))
        plt.hist(scores[labels==0], bins=50, alpha=0.6, label="Originali (label=0)", color="blue")
        plt.hist(scores[labels==1], bins=50, alpha=0.6, label="Manipolate (label=1)", color="red")
        plt.xlabel("cls_score")
        plt.ylabel("Frequenza")
        plt.title("Distribuzione score - separazione delle classi")
        plt.legend()
        
        if save_plot:
            os.makedirs(cfg.FIGURES_DIR, exist_ok=True)
            out = os.path.join(cfg.FIGURES_DIR, "score_distribution.png")
            plt.savefig(out, dpi=150, bbox_inches="tight")
            print(f"[Debug] Plot distribuzione salvato in: {out}")
            
        plt.show()


   # Istogramma confidenza predizioni corrette vs errate
    @staticmethod
    def plot_confidence_vs_errors(results, cfg, save_plot=False):
        scores = np.array(results["scores"])
        labels = np.array(results["labels"])
        preds  = (scores > 0.5).astype(int)

        tp_scores = scores[(preds == 1) & (labels == 1)]
        tn_scores = scores[(preds == 0) & (labels == 0)]
        fp_scores = scores[(preds == 1) & (labels == 0)]
        fn_scores = scores[(preds == 0) & (labels == 1)]

        print("\n--- Conteggi ---")
        print(f"  TP: {len(tp_scores)} | TN: {len(tn_scores)} "
              f"| FP: {len(fp_scores)} | FN: {len(fn_scores)}")

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        groups  = [tp_scores, tn_scores, fp_scores, fn_scores]
        titles  = ["TP (corretto -> manipolata)", "TN (corretto -> originale)",
                   "FP (errore -> predice manipolata)", "FN (errore -> predice originale)"]
        colors  = ["#55A868", "#4C72B0", "#C44E52", "#8172B2"]

        for ax, group, title, color in zip(axes.flatten(), groups, titles, colors):
            if len(group) == 0:
                ax.text(0.5, 0.5, "Nessun campione", ha="center", va="center")
            else:
                ax.hist(group, bins=30, color=color, edgecolor="white", alpha=0.85)
                ax.axvline(group.mean(), color="black", linestyle="--",
                           linewidth=1.5, label=f"media={group.mean():.3f}")
                ax.legend()
            ax.set_xlim(0, 1)
            ax.set_xlabel("cls_score")
            ax.set_ylabel("Frequenza")
            ax.set_title(f"{title}\n(n={len(group)})")

        plt.tight_layout()
        if save_plot:
            os.makedirs(cfg.FIGURES_DIR, exist_ok=True)
            out = os.path.join(cfg.FIGURES_DIR, "confidence_vs_errors.png")
            plt.savefig(out, dpi=150, bbox_inches="tight")
            print(f"[Debug] Plot confidenza/errori salvato in: {out}")
            
        plt.show()
