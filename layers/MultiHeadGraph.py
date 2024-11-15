import torch
import torch.nn as nn
import torch.nn.functional as F


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('bhle,hll->bhle', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Linear(c_in, c_out)

    def forward(self, x):
        return self.mlp(x)


class mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout


    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(1)).unsqueeze(0).to(x.device)
        d = adj.sum(2).unsqueeze(-1)
        h = x
        out = [h]
        a = adj / d
        for i in range(self.gdep):
            h = self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=-1)
        ho = self.mlp(ho)
        return ho


class multi_head_graph(nn.Module):
    def __init__(self, num_nodes, dropout, d_model, node_dim, num_heads, gdep):
        super(multi_head_graph, self).__init__()
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.layernorm = nn.LayerNorm(d_model)
        self.gelu = nn.GELU()
        self.n_heads = num_heads

        d_x = d_model // num_heads
        self.x_projection = nn.Linear(d_model, d_x * num_heads)

        self.node2vec1 = nn.Parameter(torch.randn(num_heads, num_nodes, node_dim), requires_grad=True)
        self.node2vec2 = nn.Parameter(torch.randn(num_heads, node_dim, num_nodes), requires_grad=True)

        self.mixhop = mixprop(c_in=d_x, c_out=d_x, gdep=gdep, dropout=self.dropout)

    def forward(self, x):

        M1 = F.tanh(self.node2vec1)
        M2 = F.tanh(self.node2vec2)
        adp = F.relu(F.tanh((torch.bmm(M1, M2) - torch.bmm(M2.permute(0, 2, 1), M1.permute(0, 2, 1)))))
        B, L, _ = x.shape
        H = self.n_heads
        x = self.x_projection(x).view(B, H, L, -1)
        out = self.mixhop(x, adp)
        out = self.gelu(out)
        out = out.view(B, L, -1)

        return self.layernorm(out)


class MultiHeadGraphEncoderLayer(nn.Module):
    def __init__(self, num_nodes, d_model, num_heads, node_dim, gdep, d_ff=None, dropout=0.1, activation="gelu"):
        super(MultiHeadGraphEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.multi_head_graph = multi_head_graph(num_nodes, dropout, d_model, node_dim=node_dim, num_heads=num_heads, gdep=gdep)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):

        new_x = self.multi_head_graph(x)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class MultiHeadGraphEncoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(MultiHeadGraphEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x):
        # x [B, L, D]
        for attn_layer in self.attn_layers:
            x = attn_layer(x)

        if self.norm is not None:
            x = self.norm(x)

        return x