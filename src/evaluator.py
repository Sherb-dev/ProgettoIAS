import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from sklearn.metrics import accuracy_score, roc_curve, auc
from torch.utils.data import DataLoader
 
from src.config import Config
 
#calcoliamo tutte le metriche richieste sia pixel-wise considerando la maschera esatta che sulle bounding box
class Evaluator:
    def __init__(self, model: nn.Module, cfg: Config, seg_threshold: float | None = None, cls_threshold: float | None = None):
        self.model = model.to(cfg.DEVICE)
        self.cfg = cfg
        #usiamo due soglie differenti per classificazione binaria e localizzazione
        self.seg_threshold = seg_threshold if seg_threshold is not None else cfg.SEG_THRESHOLD
        self.cls_threshold = cls_threshold if cls_threshold is not None else cfg.CLS_THRESHOLD
 
    def predict(self, image):
        self.model.eval() #imposta la rete in modalità di valutazione
        with torch.no_grad(): #spegne il calcolo dei gradienti
            seg_logits, cls_logits = self.model(image.to(self.cfg.DEVICE)) #ottiene i tensori con gli output grezzi della rete per entrambi i task
            prob_mask = torch.sigmoid(seg_logits) #trasforma i valori grezzi in probabilità tra 0 e 1 (quanto è probabile che ogni pixel sia stato manipolato)
            binary_mask = (prob_mask > self.seg_threshold).float() #applica la soglia per la segmentazione
            cls_score = torch.sigmoid(cls_logits).squeeze(1) #trasforma il tensore in un vettore unidimensionale
        return prob_mask, binary_mask, cls_score
 
 
    @staticmethod
    def _segmentation_metrics(pred_mask, true_mask, eps=1e-6):
        #usiamo questa funzione per calcolare le metriche sui pixel (quanto la maschera di segmentazione predetta coincide con quella vera)
        pred = pred_mask.flatten()
        true = true_mask.flatten()
 
        #se sia la maschera di ground truth che quella predetta sono completamente nere significa che il modello ha predetto perfettamente un'immagine non manipolata
        #in questo caso restituiamo direttamente 1 per evitare errori strani nei calcoli o divisioni per 0
        if pred.sum() == 0 and true.sum() == 0:
            return {"iou": 1.0, "f1": 1.0, "precision": 1.0, "recall": 1.0}
 
        tp = (pred * true).sum() #veri positivi
        fp = (pred * (1 - true)).sum() #falsi positivi
        fn = ((1 - pred) * true).sum() #falsi negativi
 
        #calcoliamo le metriche
        iou = (tp + eps) / (tp + fp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
 
        return {"iou": float(iou), "f1": float(f1), "precision": float(precision), "recall": float(recall)}
 
 
    @staticmethod
    def _mask_to_bbox(mask):
    #calcoliamo le bounding box delle maschere
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask.squeeze(0)
 
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return None #se la maschera non ha pixel positivi restituisce none
        return (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
 
    @staticmethod
    def _bbox_metrics(pred_mask, true_mask, eps = 1e-6):
        #calcoliamo le stesse metriche di _segmentation_metrics ma sulle bbox per valutare se il sistema riconosce la regione giusta anche se non azzecca perfettamente i bordi della maschera
        pred_bbox = Evaluator._mask_to_bbox(pred_mask)
        true_bbox = Evaluator._mask_to_bbox(true_mask)
 
        #se entrambe le maschere sono completamente nere restituisce 1
        if pred_bbox is None and true_bbox is None:
            return {"bbox_iou": 1.0, "bbox_f1": 1.0, "bbox_precision": 1.0, "bbox_recall": 1.0}
 
        #se solo una delle due maschere è completamente nera significa che non ci sono sovrapposizioni possibili quindi restituisce 0
        if pred_bbox is None or true_bbox is None:
            return {"bbox_iou": 0.0, "bbox_f1": 0.0, "bbox_precision": 0.0, "bbox_recall": 0.0}
 
        py0, py1, px0, px1 = pred_bbox #coordinate del rettangolo predetto
        ty0, ty1, tx0, tx1 = true_bbox #coordinate del rettangolo di ground truth
 
        #calcoliamo le coordinate del rettangolo di intersezione tra le due maschere
        inter_y0, inter_y1 = max(py0, ty0), min(py1, ty1)
        inter_x0, inter_x1 = max(px0, tx0), min(px1, tx1)
        inter_h = max(0, inter_y1 - inter_y0 + 1)
        inter_w = max(0, inter_x1 - inter_x0 + 1)
        inter_area = inter_h * inter_w #se l'area è 0 le maschere non si sovrappongono per niente
 
        #calcoliamo le aree totali
        pred_area = (py1 - py0 + 1) * (px1 - px0 + 1)
        true_area = (ty1 - ty0 + 1) * (tx1 - tx0 + 1)
        union_area = pred_area + true_area - inter_area
 
        #e calcoliamo le metriche applicate alle bbox
        iou = (inter_area + eps) / (union_area + eps)
        precision = (inter_area + eps) / (pred_area + eps)
        recall = (inter_area + eps) / (true_area + eps)
        f1 = (2 * inter_area + eps) / (pred_area + true_area + eps)
 
        return {"bbox_iou": float(iou), "bbox_f1": float(f1), "bbox_precision": float(precision), "bbox_recall": float(recall)}
 
    def _metric_value(self, pred_binary, gt, metric):
        #funzione per organizzare le metriche
        if metric.startswith("bbox_"):
            return self._bbox_metrics(pred_binary, gt)[metric]
        return self._segmentation_metrics(pred_binary, gt)[metric]
   
 
    #questa funzione non viene chiamata in nessun punto del codice, la chiamiamo manualmente prima di lanciare l'addestramento per poi impostare la soglia adatta in config.py
    def find_best_threshold(self, val_loader: DataLoader, metric: str = "f1") -> tuple[float, float]:
        valid_metrics = ("f1", "iou", "bbox_iou", "bbox_f1")
        assert metric in valid_metrics, f"metric deve essere una tra {valid_metrics}"
 
        #raccogliamo le maschere predette e di ground truth
        all_prob_masks = []  
        all_gt_masks = []
 
        self.model.eval()
        with torch.no_grad():
            for image, mask, label in val_loader:
                seg_logits, _ = self.model(image.to(self.cfg.DEVICE))
                prob_mask = torch.sigmoid(seg_logits)
 
                for i in range(image.size(0)):
                    if label[i].item() == 1.0:  # solo immagini manipolate
                        all_prob_masks.append(prob_mask[i].cpu().numpy())
                        all_gt_masks.append(mask[i].cpu().numpy())
 
        #proviamo tutte le soglie con un incremento di 0.05 per volta
        thresholds = np.arange(0.1, 0.91, 0.05)
        results = []
 
        for t in thresholds:
            scores = []
            for prob, gt in zip(all_prob_masks, all_gt_masks):
                binary = (prob > t).astype(np.float32)
                scores.append(self._metric_value(binary, gt, metric))
 
            mean_score = float(np.mean(scores))
            results.append((t, mean_score))
            print(f"  threshold={t:.2f} → mean_{metric}={mean_score:.4f}")
 
        #cerchiamo il massimo
        best_threshold, best_score = max(results, key=lambda x: x[1])
        print(f"\nSoglia ottimale: {best_threshold:.2f} - {metric}={best_score:.4f}")
 
        self.seg_threshold = best_threshold  # aggiorna self.seg_threshold (SOLO segmentazione) per evaluate()
        return best_threshold, best_score
 
   
    def evaluate(self, test_loader: DataLoader) -> dict:
        all_iou, all_f1, all_prec, all_rec = [], [], [], []
        all_bbox_iou, all_bbox_f1, all_bbox_prec, all_bbox_rec = [], [], [], []
        all_bbox_iou, all_bbox_f1, all_bbox_prec, all_bbox_rec = [], [], [], []
        all_labels, all_scores = [], []
 
        for image, mask, label in test_loader:
            prob_mask, binary_mask, cls_score = self.predict(image)
 
            for i in range(image.size(0)):
                if label[i].item() == 1.0:  #calcoliamo le metriche di localizzazione solo su immagini manipolate
                    pred_np = binary_mask[i].cpu().numpy()
                    gt_np = mask[i].cpu().numpy()
 
                    m = self._segmentation_metrics(pred_np, gt_np)
                    all_iou.append(m["iou"])
                    all_f1.append(m["f1"])
                    all_prec.append(m["precision"])
                    all_rec.append(m["recall"])
 
                    bb = self._bbox_metrics(pred_np, gt_np)
                    all_bbox_iou.append(bb["bbox_iou"])
                    all_bbox_f1.append(bb["bbox_f1"])
                    all_bbox_prec.append(bb["bbox_precision"])
                    all_bbox_rec.append(bb["bbox_recall"])
 
            all_labels.extend(label.numpy().tolist())
            all_scores.extend(cls_score.cpu().numpy().tolist())
 
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        roc_auc = auc(fpr, tpr)
        preds = [1 if s > self.cls_threshold else 0 for s in all_scores]
        accuracy = accuracy_score(all_labels, preds)
 
        return {
            # Metriche pixel-wise (sul contorno esatto della maschera)
            # Metriche pixel-wise (sul contorno esatto della maschera)
            "mean_iou": float(np.mean(all_iou)),
            "mean_f1": float(np.mean(all_f1)),
            "mean_precision": float(np.mean(all_prec)),
            "mean_recall": float(np.mean(all_rec)),
 
            # Metriche a livello di bounding box (sul rettangolo che racchiude la regione)
            "mean_bbox_iou": float(np.mean(all_bbox_iou)),
            "mean_bbox_f1": float(np.mean(all_bbox_f1)),
            "mean_bbox_precision": float(np.mean(all_bbox_prec)),
            "mean_bbox_recall": float(np.mean(all_bbox_rec)),
 
            # Classificazione
            "accuracy": float(accuracy),
            "roc_auc": float(roc_auc),
            "fpr": fpr,
            "tpr": tpr,
            "labels": all_labels,
            "scores": all_scores,
        }