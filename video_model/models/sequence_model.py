import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        # Handle odd dimensions cleanly
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        input_size: int = 345,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        self.out_features = hidden_size * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_size]
        if lengths is not None:
            # Packed sequence to handle variable lengths
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, (hn, cn) = self.lstm(packed_x)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            out, (hn, cn) = self.lstm(x)

        # Mean pool output across the sequence length (ignoring zero paddings if possible)
        # Using simple mean pooling across the actual sequence lengths
        if lengths is not None:
            pooled = []
            for i, length in enumerate(lengths):
                # Take average of non-padded frames
                pooled.append(out[i, :length].mean(dim=0))
            return torch.stack(pooled)
        else:
            return out.mean(dim=1)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_size: int = 345,
        d_model: int = 128,
        num_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 500
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_features = d_model

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_size]
        x = self.input_projection(x)
        x = self.pos_encoder(x)

        # Create key padding mask
        key_padding_mask = None
        if lengths is not None:
            batch_size, seq_len, _ = x.shape
            # True elements represent positions that should be ignored by attention
            key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)
            for i, length in enumerate(lengths):
                key_padding_mask[i, length:] = True

        out = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)

        # Mean pool output across non-padded sequence elements
        if lengths is not None:
            pooled = []
            for i, length in enumerate(lengths):
                pooled.append(out[i, :length].mean(dim=0))
            return torch.stack(pooled)
        else:
            return out.mean(dim=1)


class SequenceClassifier(nn.Module):
    def __init__(self, encoder_type: str = 'lstm', num_classes: int = 10, **kwargs):
        super().__init__()
        self.encoder_type = encoder_type.lower()

        if self.encoder_type == 'lstm':
            self.encoder = LSTMEncoder(**kwargs)
        elif self.encoder_type == 'transformer':
            self.encoder = TransformerEncoder(**kwargs)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        self.fc = nn.Sequential(
            nn.Dropout(p=kwargs.get('dropout', 0.2)),
            nn.Linear(self.encoder.out_features, num_classes)
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        features = self.encoder(x, lengths)
        logits = self.fc(features)
        return logits
