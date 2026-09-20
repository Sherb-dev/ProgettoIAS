import torch
import torch.nn.functional as F

def dice_loss(pred_logits, target, eps=1e-6, reduction="mean"):
    #il parametro reduction per convenzione di pytorch è impostato a mean di default
    #significa che restituisce un singolo scalare, ovvero una media dei valori della loss sul batch
    #quando reduction="none", la funzione restituisce un tensore con la loss per campione (usiamo questa impostazione)
    
    pred=torch.sigmoid(pred_logits) #trasforma gli output della rete in probabilità (tra 0 e 1)
    pred=pred.view(pred.size(0), -1) #prende le immagini 2D e le trasforma in vettori monodimensionali
    target=target.view(target.size(0), -1)
    
    intersection=(pred * target).sum(dim=1) #dim=1 perchè sommiamo tra loro i pixel di una stessa immagine
    union=pred.sum(dim=1) + target.sum(dim=1)
    dice=(2 * intersection + eps) / (union + eps) #soft dice con il parametro epsilon per evitare divisioni per 0
    loss=1-dice 

    if reduction=="none":
        return loss
    return loss.mean() 


def combined_loss(seg_logits, gt_mask, cls_logits, labels, w_seg=1.0, w_cls=0.5):
    #w_seg è il peso che stiamo assegnando al task di segmentazione, allo stesso modo w_cls è quello per il task di classificazione
    #diamo più importanza alla segmentazione, quindi alla localizzazione, perchè è il task più difficile su cui il modello deve concentrarsi maggiormente
    #senza comunque trascurare la classificazione
    
    bce_seg=F.binary_cross_entropy_with_logits(seg_logits, gt_mask)
    bce_cls=F.binary_cross_entropy_with_logits(cls_logits.squeeze(1), labels)

    dice=dice_loss(seg_logits, gt_mask, reduction="none")
    n_manip=labels.sum().clamp(min=1.0) #somma le labels quindi conta il numero di immagini manipolate, usiamo clamp a 1 per evitare divisioni per 0
    weighted_dice=(dice*labels).sum() / n_manip #moltiplica ogni valore della dice per l'etichetta vera, in modo da "spegnere" le immagini originali e considerare solo quelle manipolate

    return w_seg*(bce_seg+weighted_dice) + w_cls*bce_cls #combinazione delle due loss, usiamo entrambe per la localizzazione e solo la bce per la classificazione
