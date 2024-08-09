import numpy as np
import einops
import torch
import torch.nn as nn
from .components.pose_transformer import TransformerDecoder


class SMPLTransformerDecoderHead(nn.Module):
    """ HMR2 Cross-attention based SMPL Transformer decoder
    """
    def __init__(self, ):
        super().__init__()
        transformer_args = dict(
            depth = 6,  # originally 6
            heads = 8,
            mlp_dim = 1024,
            dim_head = 64,
            dropout = 0.0,
            emb_dropout = 0.0,
            norm = "layer",
            context_dim = 1280,
            num_tokens = 1,
            token_dim = 1,
            dim = 1024
            )
        self.transformer = TransformerDecoder(**transformer_args)

        dim = 1024
        npose = 24*6
        self.decpose = nn.Linear(dim, npose)
        self.decshape = nn.Linear(dim, 10)
        self.deccam = nn.Linear(dim, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        mean_params = np.load('data/smpl/smpl_mean_params.npz')
        init_body_pose = torch.from_numpy(mean_params['pose'].astype(np.float32)).unsqueeze(0)
        init_betas = torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32)).unsqueeze(0)
        self.register_buffer('init_body_pose', init_body_pose)
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_cam', init_cam)

        
    def forward(self, x, **kwargs):

        batch_size = x.shape[0]
        # vit pretrained backbone is channel-first. Change to token-first
        x = einops.rearrange(x, 'b c h w -> b (h w) c')

        init_body_pose = self.init_body_pose.expand(batch_size, -1)
        init_betas = self.init_betas.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        # Pass through transformer
        token = torch.zeros(batch_size, 1, 1).to(x.device)
        token_out = self.transformer(token, context=x)
        token_out = token_out.squeeze(1) # (B, C)

        # Readout from token_out
        pred_pose = self.decpose(token_out)  + init_body_pose
        pred_shape = self.decshape(token_out)  + init_betas
        pred_cam = self.deccam(token_out)  + init_cam

        return pred_pose, pred_shape, pred_cam


class temporal_attention(nn.Module):
    def __init__(self, in_dim=1280, out_dim=1280, hdim=512, nlayer=6, nhead=4, residual=False):
        super(temporal_attention, self).__init__()
        self.hdim = hdim
        self.out_dim = out_dim
        self.residual = residual
        self.l1 = nn.Linear(in_dim, hdim)
        self.l2 = nn.Linear(hdim, out_dim)

        self.pos_embedding = PositionalEncoding(hdim, dropout=0.1)
        TranLayer = nn.TransformerEncoderLayer(d_model=hdim, nhead=nhead, dim_feedforward=1024,
                                               dropout=0.1, activation='gelu')
        self.trans = nn.TransformerEncoder(TranLayer, num_layers=nlayer)
        
        nn.init.xavier_uniform_(self.l1.weight, gain=0.01)
        nn.init.xavier_uniform_(self.l2.weight, gain=0.01)

    def forward(self, x):
        x = x.permute(1,0,2)  # (b,t,c) -> (t,b,c)

        h = self.l1(x)
        h = self.pos_embedding(h)
        h = self.trans(h)
        h = self.l2(h)

        if self.residual:
            x = x[..., :self.out_dim] + h
        else:
            x = h
        x = x.permute(1,0,2)

        return x

    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)
