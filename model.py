"""Mô hình Two-Stream Fusion cho nhận diện tin giả đa phương tiện (tiểu luận, mục 3.4).

Luồng dữ liệu (B = batch, N = số token, d = 768 với bert-base):
  văn bản  --BERT-->            T      [B, N, d]
  ảnh      --ResNet-50 + Linear--> V   [B, 49, d]      (bước 2)
  Cross-Attention: Q = T, K = V, Val = V -> H [B, N, d]   (bước 3)
  z = [t_CLS ; MeanPool(H)]                [B, 2d]
  Dense -> ReLU -> Dense -> logits         [B, 2]       (bước 4; softmax nằm trong CrossEntropyLoss)
Nhãn: 0 = Fake, 1 = Real.
"""
import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    """ResNet-50 bỏ avgpool + fc -> bản đồ 7x7x2048 -> 49 véc-tơ vùng ảnh -> chiếu về d chiều."""

    def __init__(self, d_model, pretrained=True):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone = nn.Sequential(*list(net.children())[:-2])  # [B, 2048, 7, 7]
        self.proj = nn.Linear(2048, d_model)                        # W_v trong tiểu luận

    def forward(self, images):
        fmap = self.backbone(images)            # [B, 2048, 7, 7]
        fmap = fmap.flatten(2).transpose(1, 2)  # [B, 49, 2048]
        return self.proj(fmap)                  # V: [B, 49, d]


class TextEncoder(nn.Module):
    def __init__(self, name, tiny_vocab=None):
        super().__init__()
        from transformers import AutoModel, BertConfig, BertModel
        if tiny_vocab is not None:  # chỉ dùng cho --smoke (không cần tải mô hình)
            cfg = BertConfig(vocab_size=tiny_vocab, hidden_size=64, num_hidden_layers=2,
                             num_attention_heads=2, intermediate_size=128)
            self.bert = BertModel(cfg)
        else:
            self.bert = AutoModel.from_pretrained(name)
        self.hidden = self.bert.config.hidden_size

    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # T


class TwoStreamMFND(nn.Module):
    """fusion: 'cross' (đề xuất) | 'concat' (A3) | 'text_only' (A4) | 'image_only' (A5)."""

    def __init__(self, text_model="bert-base-chinese", fusion="cross", pretrained_image=True,
                 n_heads=1, dropout=0.1, hidden=256, tiny_vocab=None):
        super().__init__()
        assert fusion in ("cross", "concat", "text_only", "image_only")
        self.fusion = fusion
        if fusion == "image_only":
            d = 64 if tiny_vocab is not None else 768
            self.text = None
        else:
            self.text = TextEncoder(text_model, tiny_vocab)
            d = self.text.hidden
        self.image = None if fusion == "text_only" else ImageEncoder(d, pretrained_image)
        if fusion == "cross":
            # n_heads=1 khớp đúng công thức softmax(QK^T/sqrt(d_k))V trong tiểu luận.
            self.cross = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        in_dim = 2 * d if fusion in ("cross", "concat") else d
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, input_ids, attention_mask, images=None, return_attn=False):
        attn = None
        if self.fusion == "image_only":
            z = self.image(images).mean(1)
        else:
            T = self.text(input_ids, attention_mask)   # [B, N, d]
            t_cls = T[:, 0]                            # véc-tơ [CLS]
            if self.fusion == "text_only":
                z = t_cls
            else:
                V = self.image(images)                 # [B, 49, d]
                if self.fusion == "concat":
                    z = torch.cat([t_cls, V.mean(1)], dim=-1)
                else:
                    # truy vấn từ token văn bản, khóa/giá trị từ 49 vùng ảnh
                    H, attn = self.cross(query=T, key=V, value=V, need_weights=True)  # attn: [B, N, 49]
                    m = attention_mask.unsqueeze(-1).to(H.dtype)                       # bỏ qua token đệm
                    pooled = (H * m).sum(1) / m.sum(1).clamp(min=1)                    # MeanPool(H)
                    z = torch.cat([t_cls, pooled], dim=-1)
        logits = self.head(z)
        return (logits, attn) if return_attn else logits
