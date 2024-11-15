from layers.Embed import DataEmbedding_inverted
from layers.MultiHeadGraph import *
from layers.StandardNorm import Normalize

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_DI = configs.use_DI
        # Embedding

        if self.use_DI:
            self.enc_embedding = nn.ModuleList(
                [DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                        configs.dropout)
                 for i in range(2)])

            self.normalize_layers = torch.nn.ModuleList(
                [
                    Normalize(configs.enc_in, affine=configs.affine, non_norm=True if configs.use_norm == 0 else False)
                    for i in range(2)
                ]
            )
        else:
            self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                        configs.dropout)
            self.normalize_layer = Normalize(configs.enc_in, affine=configs.affine,
                                             non_norm=True if configs.use_norm == 0 else False)

        # Encoder
        self.encoder = MultiHeadGraphEncoder(
            [
                MultiHeadGraphEncoderLayer(configs.enc_in, configs.d_model, configs.n_heads, configs.node_dim, configs.gdep, configs.d_ff) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Decoder
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # list
        enc_out_list = []
        _, _, N = x_enc.shape
        if self.use_DI:
            x_enc = self.__diff(x_enc, x_mark_enc, padding_method='0_begin')
            for i, x in zip(range(len(x_enc)), x_enc, ):
                x = self.normalize_layers[i](x, 'norm')
                enc_out_i = self.enc_embedding[i](x, None) # (B, D, d_model)
                enc_out_list.append(enc_out_i)
            enc_out = self.configs.seita * enc_out_list[0] + (1 - self.configs.seita) * enc_out_list[1]
        else:
            x_enc = self.normalize_layer(x_enc, 'norm')
            enc_out = self.enc_embedding(x_enc, None)

        enc_out = self.encoder(enc_out)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_DI:
            dec_out = self.normalize_layers[0](dec_out, 'denorm')
        else:
            dec_out = self.normalize_layer(dec_out, 'denorm')

        return dec_out[:, -self.pred_len:, :]


    def __diff(self, x_enc, x_mark_enc, padding_method):

        x_diff = x_enc[:, 1:, :] - x_enc[:, :-1, :]
        if padding_method == 'repeat_begin':
            pad = x_diff[:, 0:1, :]
        elif padding_method == '0_begin':
            pad = torch.zeros_like(x_diff[:, 0:1, :])
        elif padding_method == 'repeat_end':
            pad = x_diff[:, -1:, :]
        elif padding_method == '0_end':
            pad = torch.zeros_like(x_diff[:, -1:, :])
        else:
            raise ValueError("Unsupported padding method.")

        if 'begin' in padding_method:
            x_diff_padded = torch.cat([pad, x_diff], dim=1)
        else:
            x_diff_padded = torch.cat([x_diff, pad], dim=1)

        x_enc_list = [x_enc, x_diff_padded]

        return x_enc_list
