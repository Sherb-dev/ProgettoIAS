import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF

#blocco di base
class ConvBlock_old(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #due strati convoluzionali in sequenza
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), #convoluzione per estrarre le feature
            nn.BatchNorm2d(out_channels), #normalizzazione
            nn.ReLU(inplace=True), #funzione di attivazione

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout_prob > 0.0:
            layers.append(nn.Dropout2d(dropout_prob)) # Dropout2d per feature map convoluzionali
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #applica la convoluzione per estrarre le feature specifiche che verranno passate direttamente al decoder tramite le skip connections 
        self.conv = ConvBlock(in_channels, out_channels) 
        #dimezza le dimensioni dell'immagine, che verrà poi passata al blocco successivo, per fare il modo che il modello apprenda il contesto globale
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        skip = self.conv(x)
        down = self.pool(skip)
        return skip, down

class AttentionGate(nn.Module):
    def __init__(self, g_channels, s_channels, out_channels):
        super().__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(g_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(s_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, g, s):
        g1 = self.Wg(g) #l'attention gate riceve le feature di basso livello dal decoder (gating signal)
        s1 = self.Ws(s) #e quelle di alto livello dall'encoder tramite le skip connections
        out = F.relu(g1 + s1) #unisce i due segnali
        psi = self.psi(out) #e applica la sigmoid per creare le attention maps
        return s * psi, psi #restituisce i dettagli filtrati (i pixel moltiplicati per la probabilità che siano manipolati o meno) e le attention map

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_attention=False):
        super().__init__()
        self.use_attention=use_attention
        if use_attention:
            self.att = AttentionGate(in_channels, skip_channels, out_channels)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)
  
    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True) #ingrandisce l'immagine proveniente dal livello inferiore
        attention_map=None 
        if self.use_attention:
            skip, attention_map = self.att(x, skip) #filtra la skip connection con l'attention
        x = torch.cat([x, skip], dim=1) #concatena l'immagine appena ricevuta e ingrandita con in risultati delle skip, con o senza attention
        return self.conv(x), attention_map #fonde e restituisce l'input attraverso le convoluzioni

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=None, use_attention=False):
        super().__init__()
        if base_channels is None:
            from src.config import Config  #import locale per evitare dipendenze circolari
            base_channels = Config.BASE_CHANNELS
        c1, c2, c3, c4 = base_channels, base_channels*2, base_channels*4, base_channels*8 #filtri

        #encoder
        self.enc1 = EncoderBlock(in_channels, c1)
        self.enc2 = EncoderBlock(c1, c2)
        self.enc3 = EncoderBlock(c2, c3)

        # bottleneck con dropout
        self.bottleneck = ConvBlock(c3, c4, dropout_prob=0.2)

        # decoder con dropout nel primo stadio
        self.dec1 = DecoderBlock(c4, c3, c3, use_attention=use_attention) # puoi passare dropout_prob anche qui
        self.dec2 = DecoderBlock(c3, c2, c2, use_attention=use_attention)
        self.dec3 = DecoderBlock(c2, c1, c1, use_attention=use_attention)

        #testa di segmentazione
        self.final_conv = nn.Conv2d(c1, out_channels, kernel_size=1) #comprime l'output a un singolo canale per ottenere le maschere binarie

        #testa di classificazione
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), #global average pooling per calcolare una media di tutto
            nn.Flatten(),            
            nn.Linear(c4, 128), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)          
        )

        self.use_attention=use_attention

        # Salva le attention map degli ultimi 3 decoder block
        # Utili per visualizzare quale regione è stata enfatizzata dall'attention gate
        self._last_attention_maps=[]

    def forward(self, x):
        #encoder
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)

        #bottleneck
        b = self.bottleneck(p3)

        #testa di classificazione, direttamente sull'output del bottleneck prima del decoder in modo da considerare dati più generici
        cls_logits = self.cls_head(b)

        #decoder
        d1, a1 = self.dec1(b, s3)
        d2, a2 = self.dec2(d1, s2)
        d3, a3 = self.dec3(d2, s1)

        #testa di segmentazione
        seg_logits = self.final_conv(d3)
        self._last_attention_maps=[a1, a2, a3]

        return seg_logits, cls_logits

    def get_attention_maps(self):
        return self._last_attention_maps

#creiamo due sottoclassi separate in modo da non dover modificare parametri (la flag per l'attention)
#possiamo direttamente chiamare la rete che ci interessa
class AttentionUNet(UNet):
    def __init__(self, in_channels=3, out_channels=1, base_channels=None):
        super().__init__(in_channels, out_channels, base_channels, use_attention=True)

class PlainUNet(UNet):
    def __init__(self, in_channels=3, out_channels=1, base_channels=None):
        super().__init__(in_channels, out_channels, base_channels, use_attention=False)


def count_parameters(model: nn.Module):
    #conteggio dei parametri per essere sicuri al 100% che l'unica differenza sta nell'architettura e non nei parametri
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
