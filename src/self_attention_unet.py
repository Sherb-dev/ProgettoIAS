#Plain UNet vs Self-Attention UNet

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)     # Before pooling - for skip connection
        down = self.pool(skip)  # After pooling - goes deeper
        return skip, down


class SelfAttention(nn.Module):
    """
    Self-Attention non-local (Zhang et al., 2019 — SAGAN). Lavora su UN solo
    tensore (la skip connection), mettendo in relazione ogni posizione
    spaziale con tutte le altre. Non riceve alcun segnale dal decoder: a
    differenza dell'Attention Gate, non decide "questa skip è rilevante per
    quello che sto ricostruendo ora?", ma "quali punti di QUESTA mappa sono
    correlati tra loro?" — è una dipendenza globale interna alla feature map,
    non una selezione guidata dall'esterno.

    Costo: la matrice di attenzione è N x N con N = altezza*larghezza della
    mappa. Cresce quadraticamente con la risoluzione — motivo per cui in
    questa architettura viene applicata solo al livello più profondo del
    decoder (vedi UNet più sotto).
    """

    def __init__(self, in_channels, max_positions: int = 8192):
        super().__init__()
        # max_positions: limite di sicurezza su N = H*W. Con 8192 la mappa può
        # arrivare fino a circa 90x90 prima che il modulo si rifiuti di
        # procedere. Serve a trasformare un OOM silenzioso (la GPU si
        # blocca senza un messaggio chiaro) in un errore esplicito e
        # comprensibile. Alzalo solo se sai quanta memoria hai a disposizione.
        self.max_positions = max_positions

        # Query e Key proiettano in uno spazio ridotto (1/8 dei canali): è la
        # scelta standard di SAGAN, riduce il costo di calcolo di query/key
        # senza perdere troppa capacità rappresentativa. Value resta a piena
        # dimensionalità, perché è il contenuto che verrà effettivamente
        # ridistribuito in base ai pesi di attenzione.
        reduced = max(in_channels // 8, 1)
        self.query_conv = nn.Conv2d(in_channels, reduced, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, reduced, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # gamma parte da 0: a inizio training il modulo è un'identità pura
        # (output = input), e la rete impara gradualmente QUANTO pesare il
        # contributo dell'attenzione globale. Senza questo trucco, un modulo
        # di self-attention non allenato inietterebbe rumore casuale fin
        # dalla prima iterazione, rendendo il training iniziale instabile.
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        n = height * width

        if n > self.max_positions:
            raise RuntimeError(
                f"SelfAttention: mappa {height}x{width} (N={n}) oltre il limite "
                f"di sicurezza ({self.max_positions}). La matrice di attenzione "
                f"costerebbe N^2={n * n:,} elementi -> rischio concreto di OOM. "
                f"Applica la self-attention solo a livelli più profondi (risoluzione "
                f"più bassa), oppure alza max_positions consapevolmente."
            )

        # Appiattisce le due dimensioni spaziali (H,W) in un unico asse N,
        # cosi' ogni posizione diventa un "token" che puo' essere confrontato
        # con tutti gli altri, esattamente come in un Transformer.
        query = self.query_conv(x).view(batch_size, -1, n).permute(0, 2, 1)  # [B, N, C//8]
        key = self.key_conv(x).view(batch_size, -1, n)                       # [B, C//8, N]
        value = self.value_conv(x).view(batch_size, -1, n)                   # [B, C, N]

        # Similarita' tra ogni coppia di posizioni (i, j): quanto la posizione
        # i dovrebbe "guardare" la posizione j. Softmax lungo l'ultima
        # dimensione normalizza i pesi di ogni riga a somma 1.
        energy = torch.bmm(query, key)                    # [B, N, N]
        attention = torch.softmax(energy, dim=-1)

        # Ricombina il contenuto (value) usando i pesi di attenzione: ogni
        # posizione diventa una media pesata di TUTTE le posizioni della mappa,
        # non solo del suo vicinato locale come farebbe una convoluzione 3x3.
        out = torch.bmm(value, attention.permute(0, 2, 1))  # [B, C, N]
        out = out.view(batch_size, channels, height, width)

        out = self.gamma * out + x   # connessione residua
        return out, attention


class DecoderBlock(nn.Module):
    """
    use_attention=False -> skip passata cosi' com'e' (Plain UNet)
    use_attention=True  -> skip filtrata dalla Self-Attention prima della concat
    Upsampling, concat e ConvBlock finale sono identici nelle due varianti:
    l'unica differenza dichiarata e' il trattamento della skip connection.
    """

    def __init__(self, in_channels, skip_channels, out_channels, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            self.att = SelfAttention(skip_channels)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        attn_map = None
        if self.use_attention:
            skip, attn_map = self.att(skip)   # dipende SOLO dalla skip, non da x

        x = torch.cat([x, skip], dim=1)
        return self.conv(x), attn_map


class UNet(nn.Module):
    """
    Stessa architettura, stessi canali, stessa testa di classificazione della
    PlainUNet. L'unica differenza e' `use_attention`, che decide se applicare
    la Self-Attention sulle skip connection.

    self_attention_levels: tupla di 3 bool per (dec1, dec2, dec3). Default
    (True, False, False) -> la self-attention e' applicata SOLO al livello
    piu' profondo del decoder (dec1, che riceve la skip s3).

    Perche' solo li': con IMAGE_SIZE=256 e un encoder a 3 livelli, le skip
    hanno risoluzione s1=256x256, s2=128x128, s3=64x64. Il costo della
    self-attention e' O(N^2) con N=altezza*larghezza:
        s3 (64x64,  N=4.096)   -> N^2 ≈ 16,7 milioni   -> gestibile
        s2 (128x128, N=16.384) -> N^2 ≈ 268 milioni     -> gia' pesante
        s1 (256x256, N=65.536) -> N^2 ≈ 4,3 miliardi    -> OOM quasi certo
    Applicarla a s1 o s2 di default renderebbe il training inutilizzabile
    sulla maggior parte delle GPU. Il parametro resta comunque configurabile,
    per chi vuole sperimentare consapevolmente su hardware con più memoria.
    """

    def __init__(self, in_channels=3, out_channels=1, base_channels=None,
                 use_attention=False, self_attention_levels=(True, False, False)):
        super().__init__()

        if base_channels is None:
            from src.config import Config  # import locale: evita dipendenze circolari
            base_channels = Config.BASE_CHANNELS

        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        # Encoder
        self.enc1 = EncoderBlock(in_channels, c1)
        self.enc2 = EncoderBlock(c1, c2)
        self.enc3 = EncoderBlock(c2, c3)

        # Bottleneck
        self.bottleneck = ConvBlock(c3, c4)

        # Decoder — ogni livello usa la self-attention solo se sia
        # use_attention=True SIA il livello e' abilitato in self_attention_levels
        self.dec1 = DecoderBlock(c4, c3, c3, use_attention=(use_attention and self_attention_levels[0]))
        self.dec2 = DecoderBlock(c3, c2, c2, use_attention=(use_attention and self_attention_levels[1]))
        self.dec3 = DecoderBlock(c2, c1, c1, use_attention=(use_attention and self_attention_levels[2]))

        # Segmentation head
        self.final_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

        # Classification head (global average pooling sul bottleneck)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # [B, c4, 1, 1]
            nn.Flatten(),              # [B, c4]
            nn.Linear(c4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)          # logit binario
        )

        self.use_attention = use_attention
        self._last_attention_maps = []
        # Le matrici di attenzione NON vengono conservate di default: per la
        # self-attention sono potenzialmente grandi (N x N per campione), e
        # tenerle sempre in memoria durante il training avrebbe un costo
        # inutile. Si attivano esplicitamente solo per l'analisi qualitativa.
        self.store_attention_maps = False

    def forward(self, x):
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        b = self.bottleneck(p3)

        # Classification branch (dal bottleneck, prima del decoder)
        cls_logits = self.cls_head(b)

        # Segmentation branch
        d1, a1 = self.dec1(b, s3)
        d2, a2 = self.dec2(d1, s2)
        d3, a3 = self.dec3(d2, s1)
        seg_logits = self.final_conv(d3)   # logits grezzi, sigmoid applicata altrove

        if self.store_attention_maps:
            self._last_attention_maps = [a1, a2, a3]

        return seg_logits, cls_logits

    def get_attention_maps(self):
        if not self.use_attention:
            raise ValueError("Questo modello non usa self-attention: nessuna mappa disponibile.")
        if not self.store_attention_maps:
            raise RuntimeError(
                "Imposta model.store_attention_maps = True PRIMA del forward "
                "da ispezionare (idealmente su un singolo campione, in eval)."
            )
        return self._last_attention_maps


class PlainUNet(UNet):
    """Nessuna attention: identica architettura di base, skip non filtrata."""

    def __init__(self, in_channels=3, out_channels=1, base_channels=None):
        super().__init__(in_channels, out_channels, base_channels, use_attention=False)


class SelfAttentionUNet(UNet):
    """Stessa architettura della PlainUNet, con Self-Attention sulle skip
    connection al posto degli Attention Gate. Di default attiva solo sul
    livello piu' profondo del decoder (vedi spiegazione nella docstring di UNet)."""

    def __init__(self, in_channels=3, out_channels=1, base_channels=None,
                 self_attention_levels=(True, False, False)):
        super().__init__(in_channels, out_channels, base_channels,
                          use_attention=True, self_attention_levels=self_attention_levels)


def count_parameters(model: nn.Module) -> dict:
    """Conteggio parametri totali/allenabili: da riportare nel report per
    dichiarare che la differenza tra i modelli non e' (solo) dimensionale."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}