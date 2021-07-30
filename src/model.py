import torch
import torch.nn as nn
import torch.nn.functional as F


class MalConv(nn.Module):
    def __init__(self, input_length=2000000, window_size=500):
        super(MalConv, self).__init__()

        self.embed = nn.Embedding(257, 8, padding_idx=0)

        self.conv_1 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)
        self.conv_2 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)

        self.BatchNorm1d = nn.BatchNorm1d(128)

        self.pooling = nn.MaxPool1d(int(input_length / window_size))

        self.fc_1 = nn.Linear(128, 128)
        self.fc_2 = nn.Linear(128, 9)

        # self.BatchNorm1d = nn.BatchNorm1d(128)

        self.sigmoid = nn.Sigmoid()

    # self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.embed(x)
        # Channel first
        x = torch.transpose(x, -1, -2)

        cnn_value = self.conv_1(x.narrow(-2, 0, 4))
        cnn_value = self.BatchNorm1d(cnn_value)
        gating_weight = self.sigmoid(self.conv_2(x.narrow(-2, 4, 4)))

        x = cnn_value * gating_weight
        x = self.pooling(x)

        x = x.view(-1, 128)
        x = self.fc_1(x)
        # x = self.BatchNorm1d(x)
        x = self.fc_2(x)
        # x = self.sigmoid(x)

        return x


class PreMalConv(nn.Module):
    """
    Architecture implementation.


    def __init__(self, pretrained_path=None, embedding_size=8, max_input_size=2 ** 20):
            super(MalConv, self).__init__(embedding_size, max_input_size, 256, False)
            self.embedding_1 = nn.Embedding(num_embeddings=257, embedding_dim=embedding_size)
            self.conv1d_1 = nn.Conv1d(in_channels=embedding_size, out_channels=128, kernel_size=(500,), stride=(500,),
                                                              groups=1, bias=True)
            self.conv1d_2 = nn.Conv1d(in_channels=embedding_size, out_channels=128, kernel_size=(500,), stride=(500,),
                                                              groups=1, bias=True)
            self.dense_1 = nn.Linear(in_features=128, out_features=128, bias=True)
            self.dense_2 = nn.Linear(in_features=128, out_features=1, bias=True)
    """

    def __init__(self, input_length=2 ** 20, window_size=500):
        super(PreMalConv, self).__init__()

        self.embedding_1 = nn.Embedding(257, 8, padding_idx=0)

        self.conv1d_1 = nn.Conv1d(8, 128, window_size, stride=window_size, bias=True)
        self.conv1d_2 = nn.Conv1d(8, 128, window_size, stride=window_size, bias=True)

        self.pooling = nn.MaxPool1d(int(input_length / window_size))

        self.dense_1 = nn.Linear(128, 128)
        self.dense_2 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding_1(x)
        # Channel first
        x = torch.transpose(x, -1, -2)

        cnn_value = self.conv1d_1(x)
        cnn_value = torch.relu(cnn_value)
        gating_weight = self.sigmoid(self.conv1d_2(x))

        x = cnn_value * gating_weight

        global_max_pooling1d_1 = F.max_pool1d(input=x, kernel_size=x.size()[2:])
        global_max_pooling1d_1_flatten = global_max_pooling1d_1.view(
            global_max_pooling1d_1.size(0), -1
        )

        x = torch.relu(self.dense_1(global_max_pooling1d_1_flatten))
        dense_1_activation = torch.relu(x)
        dense_2 = self.dense_2(x)
        dense_2_activation = torch.sigmoid(dense_2)

        return dense_2_activation


class MalConvBase(nn.Module):
    def __init__(self, embed_dim, max_len, out_channels, window_size, dropout=0.5):
        super(MalConvBase, self).__init__()
        self.embed = nn.Embedding(257, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=out_channels * 2,
            kernel_size=window_size,
            stride=window_size,
        )
        self.fc = nn.Linear(out_channels, 1)

    def forward(self, x):
        embedding = self.dropout(self.embed(x))
        conv_in = embedding.permute(0, 2, 1)
        conv_out = self.conv(conv_in)
        glu_out = F.glu(conv_out, dim=1)
        values, _ = glu_out.max(dim=-1)
        output = self.fc(values)# .squeeze(1)
        return output


class MalConvPlus(nn.Module):
    def __init__(self, embed_dim, max_len, out_channels, window_size, dropout=0.5):
        super(MalConvPlus, self).__init__()
        self.tok_embed = nn.Embedding(257, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=out_channels * 2,
            kernel_size=window_size,
            stride=window_size,
        )
        self.fc = nn.Linear(out_channels, 1)

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        tok_embedding = self.tok_embed(x)
        pos = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to(x.device)
        pos_embedding = self.pos_embed(pos)
        embedding = self.dropout(tok_embedding + pos_embedding)
        conv_in = embedding.permute(0, 2, 1)
        conv_out = self.conv(conv_in)
        glu_out = F.glu(conv_out, dim=1)
        values, _ = glu_out.max(dim=-1)
        output = self.fc(values)# .squeeze(1)
        return output


class RCNN(nn.Module):
    def __init__(
        self,
        embed_dim,
        out_channels,
        window_size,
        module,
        hidden_size,
        num_layers,
        bidirectional,
        residual,
        dropout=0.5,
    ):
        super(RCNN, self).__init__()
        assert module.__name__ in {
            "RNN",
            "GRU",
            "LSTM",
        }, "`module` must be a `torch.nn` recurrent layer"
        self.residual = residual
        self.embed = nn.Embedding(257, embed_dim)
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=out_channels,
            kernel_size=window_size,
            stride=window_size,
        )
        self.rnn = module(
            input_size=out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        rnn_out_size = (int(bidirectional) + 1) * hidden_size
        if residual:
            self.fc = nn.Linear(out_channels + rnn_out_size, 1)
        else:
            self.fc = nn.Linear(rnn_out_size, 1)

    def forward(self, x):
        embedding = self.dropout(self.embed(x))
        conv_in = embedding.permute(0, 2, 1)
        conv_out = self.conv(conv_in)
        if self.residual:
            values, _ = conv_out.max(dim=-1)
        conv_out = conv_out.permute(2, 0, 1)
        rnn_out, _ = self.rnn(conv_out)
        fc_in = rnn_out[-1]
        if self.residual:
            fc_in = torch.cat((fc_in, values), dim=-1)
        output = self.fc(fc_in).squeeze(1)
        return output


class AttentionRCNN(nn.Module):
    def __init__(
        self,
        embed_dim,
        out_channels,
        window_size,
        module,
        hidden_size,
        num_layers,
        bidirectional,
        attn_size,
        residual,
        dropout=0.5,
    ):
        super(AttentionRCNN, self).__init__()
        assert module.__name__ in {
            "RNN",
            "GRU",
            "LSTM",
        }, "`module` must be a `torch.nn` recurrent layer"
        self.residual = residual
        self.embed = nn.Embedding(257, embed_dim)
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=out_channels,
            kernel_size=window_size,
            stride=window_size,
        )
        self.rnn = module(
            input_size=out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )
        rnn_out_size = (int(bidirectional) + 1) * hidden_size
        self.local2attn = nn.Linear(rnn_out_size, attn_size)
        self.global2attn = nn.Linear(rnn_out_size, attn_size, bias=False)
        self.attn_scale = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(attn_size, 1))
        )
        self.dropout = nn.Dropout(dropout)
        if residual:
            self.fc = nn.Linear(out_channels + rnn_out_size, 1)
        else:
            self.fc = nn.Linear(rnn_out_size, 1)

    def forward(self, x):
        embedding = self.dropout(self.embed(x))
        conv_in = embedding.permute(0, 2, 1)
        conv_out = self.conv(conv_in)
        if self.residual:
            values, _ = conv_out.max(dim=-1)
        conv_out = conv_out.permute(2, 0, 1)
        rnn_out, _ = self.rnn(conv_out)
        global_rnn_out = rnn_out.mean(dim=0)
        attention = torch.tanh(
            self.local2attn(rnn_out) + self.global2attn(global_rnn_out)
        ).permute(1, 0, 2)
        alpha = F.softmax(attention.matmul(self.attn_scale), dim=-1)
        rnn_out = rnn_out.permute(1, 0, 2)
        fc_in = (alpha * rnn_out).sum(dim=1)
        if self.residual:
            fc_in = torch.cat((fc_in, values), dim=-1)
        output = self.fc(fc_in).squeeze(1)
        return output
