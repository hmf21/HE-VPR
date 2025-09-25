import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Identity, Parameter, init


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Classifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        ##(B, L, 1+14*14, C)  => (B, C)
        x = x[:, -1, 0, :]
        return self.fc(x)


class Tokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=0,
                 n_input_channels=3,
                 n_output_channels=64):
        super(Tokenizer, self).__init__()

        self.conv_layer = nn.Conv2d(n_input_channels, n_output_channels,
                                    kernel_size=(kernel_size, kernel_size),
                                    stride=(stride, stride),
                                    padding=(padding, padding), bias=True)

        self.flattener = nn.Flatten(2, 3)
        self.apply(self.init_weight)

    def forward(self, x):
        return self.flattener(self.conv_layer(x)).transpose(-2, -1)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class TransformerEncoder(Module):
    def __init__(self,
                 embedding_dim=256,
                 num_layers=6,
                 num_heads=4,
                 mlp_ratio=2,
                 dropout_rate=0.1,
                 attention_dropout=0.1,
                 stochastic_depth_rate=0.1,
                 positional_embedding=False,
                 sequence_length=None,
                 cls_token=False):
        super().__init__()
        self.cls_token = cls_token
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim

        if cls_token:
            self.class_emb = Parameter(torch.zeros(1, 1, self.embedding_dim),
                                       requires_grad=True)
        ### TODO
        if positional_embedding:
            assert sequence_length != None
            self.positional_emb = Parameter(self.sinusoidal_embedding( \
                sequence_length, embedding_dim), requires_grad=False)
        ### ###

        self.dropout = Dropout(p=dropout_rate)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_layers)]
        self.blocks = ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout_rate,
                                    attention_dropout=attention_dropout, drop_path_rate=dpr[i])
            for i in range(num_layers)])
        self.norm = LayerNorm(embedding_dim)

        self.apply(self.init_weight)

    def forward(self, x):

        if self.cls_token:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        x = self.dropout(x)
        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            features.append(self.norm(x))
        return torch.stack(features, -3)

    @staticmethod
    def init_weight(m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)


class Attention(Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(dim, dim * 3, bias=False)
        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_norm = LayerNorm(d_model)
        self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()

        self.activation = F.gelu

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Extractor(nn.Module):
    def __init__(self,
                 embedding_dim=256,
                 n_input_channels=3,
                 token_dim_reduce=4,
                 *args, **kwargs):
        super(Extractor, self).__init__()
        self.embedding_dim = embedding_dim
        self.token_dim_reduce = token_dim_reduce

        self.conv1 = self.__build_conv(3, 64)  # (64, 112,112) **
        self.conv2 = self.__build_conv(64, 128)  # (128, 56, 56) **
        self.conv3 = self.__build_conv(128, 256)  # (256, 28, 28) **
        self.conv4 = self.__build_conv(256, 512)  # (512, 14, 14) **

        self.apply(self.init_weight)

        self.tokenizer1 = self.__build_tokenizer(64, 8)
        self.tokenizer2 = self.__build_tokenizer(128, 4)
        self.tokenizer3 = self.__build_tokenizer(256, 2)
        self.tokenizer4 = self.__build_tokenizer(512, 1)
        # (B, 14*14, C)

        self.transformer = self.__build_transformer(*args, **kwargs)

    def forward(self, x):
        map1 = self.conv1(x)
        map2 = self.conv2(map1)
        map3 = self.conv3(map2)
        map4 = self.conv4(map3)

        # (B, C)
        seq1 = self.tokenizer1(map1)
        seq2 = self.tokenizer2(map2)
        seq3 = self.tokenizer3(map3)
        seq4 = self.tokenizer4(map4)

        seq = torch.cat((seq1, seq2, seq3, seq4), dim=-1)  # (B, 14*14, C*4)

        return self.transformer(seq)  # (B, L, 1+14*14, C)

    def __build_conv(self, in_dim, out_dim):
        return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1,
                                       padding=1, bias=True),
                             nn.BatchNorm2d(out_dim),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def __build_tokenizer(self, n_channels, patch_size):
        return Tokenizer(n_input_channels=n_channels,
                         n_output_channels=self.embedding_dim // self.token_dim_reduce,
                         kernel_size=patch_size, stride=patch_size, padding=0)

    def __build_transformer(self, *args, **kwargs):
        return TransformerEncoder(embedding_dim=self.embedding_dim,
                                  cls_token=True,
                                  dropout_rate=0.1,  # => 0.1
                                  attention_dropout=0.1,
                                  stochastic_depth_rate=0.1,
                                  *args, **kwargs)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)


def extract_transvpr():
    return Extractor(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256)
