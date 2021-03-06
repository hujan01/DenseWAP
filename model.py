'''
Author: sigmoid
Description: Encoder-Decoder
Email: 595495856@qq.com
Date: 2021-02-21 14:20:39
LastEditTime: 2021-02-21 14:25:26
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Encoder params
depth = 16
growth_rate = 24

# Decoder params
n = 256
n_prime = 512
decoder_conv_filters = 256
embedding_dim = 256

class BottleneckBlock(nn.Module):
    """
    Dense Bottleneck Block
    It contains two convolutional layers, a 1x1 and a 3x3.
    """

    def __init__(self, input_size, growth_rate, dropout_rate=0.2):
        """
        Args:
            input_size (int): Number of channels of the input
            growth_rate (int): Number of new features being added. That is the ouput
                size of the last convolutional layer.
            dropout_rate (float, optional): Probability of dropout [Default: 0.2]
        """
        super(BottleneckBlock, self).__init__()
        inter_size = 4*growth_rate

        self.relu = nn.ReLU(inplace=True)
        self.norm1 = nn.BatchNorm2d(inter_size)
        self.conv1 = nn.Conv2d(
            input_size, inter_size, kernel_size=1, stride=1, bias=False
        )
        self.norm2 = nn.BatchNorm2d(growth_rate)
        self.conv2 = nn.Conv2d(
            inter_size, growth_rate, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.norm1(self.relu(self.conv1(x)))
        out = self.norm2(self.relu(self.conv2(out)))
        out = self.dropout(out)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    """
    Transition Block
    A transition layer reduces the number of feature maps in-between two bottleneck
    blocks.
    """

    def __init__(self, input_size, output_size):
        """
        Args:
            input_size (int): Number of channels of the input
            output_size (int): Number of channels of the output
        """
        super(TransitionBlock, self).__init__()
        self.norm = nn.BatchNorm2d(output_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            input_size, output_size, kernel_size=1, stride=1, bias=False
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.norm(self.relu(self.conv(x)))
        return self.pool(out)
 
class DenseBlock(nn.Module):
    """
    Dense block
    A dense block stacks several bottleneck blocks.
    """

    def __init__(self, input_size, growth_rate, depth, dropout_rate=0.2):
        """
        Args:
            input_size (int): Number of channels of the input
            growth_rate (int): Number of new features being added per bottleneck block
            depth (int): Number of bottleneck blocks
            dropout_rate (float, optional): Probability of dropout [Default: 0.2]
        """
        super(DenseBlock, self).__init__()
        layers = [
            BottleneckBlock(
                input_size + i * growth_rate, growth_rate, dropout_rate=dropout_rate
            )
            for i in range(depth)
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):
    """Multi-scale Dense Encoder
    A multi-scale dense encoder with two branches. The first branch produces
    low-resolution annotations, as a regular dense encoder would, and the second branch
    produces high-resolution annotations.
    """
    def __init__(
        self, img_channels=1, num_in_features=48, dropout_rate=0.2, checkpoint=None
    ):
        """
        Args:
            img_channels (int, optional): Number of channels of the images [Default: 1]
            num_in_features (int, optional): Number of channels that are created from
                the input to feed to the first dense block [Default: 48]
            dropout_rate (float, optional): Probability of dropout [Default: 0.2]
            checkpoint (dict, optional): State dictionary to be loaded
        """
        super(Encoder, self).__init__()
        self.conv0 = nn.Conv2d(
            img_channels,
            num_in_features,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.norm0 = nn.BatchNorm2d(num_in_features)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        num_features = num_in_features
        self.block1 = DenseBlock(
            num_features,
            growth_rate=growth_rate,
            depth=depth,
            dropout_rate=dropout_rate,
        )

        num_features = num_features + depth * growth_rate
        self.trans1 = TransitionBlock(num_features, num_features // 2)
        num_features = num_features // 2
        self.block2 = DenseBlock(
            num_features,
            growth_rate=growth_rate,
            depth=depth,
            dropout_rate=dropout_rate,
        )

        num_features = num_features + depth * growth_rate
        self.trans2 = TransitionBlock(num_features, num_features // 2)
        num_features = num_features // 2
        self.block3 = DenseBlock(
            num_features,
            growth_rate=growth_rate,
            depth=depth,
            dropout_rate=dropout_rate,
        )

    def forward(self, x):
        out = self.conv0(x)
        out = self.relu(self.norm0(out))
        out = self.max_pool(out)
        out = self.block1(out)
        out = self.trans1(out)
        out = self.block2(out)
        out = self.trans2(out)
        out = self.block3(out)
        return out

class CoverageAttention(nn.Module):
    """Coverage attention
    The coverage attention is a multi-layer perceptron, which takes encoded annotations
    and creates a context vector.
    """
    def __init__(
        self,
        input_size,
        output_size,
        kernel_size,
        padding=0,
        device=device,
    ):
        """
        Args:
            input_size (int): Number of channels of the input
            output_size (int): Number of channels of the coverage
            attn_size (int): Length of the annotation vector
            kernel_size (int): Kernel size of the 1D convolutional layer
            padding (int, optional): Padding of the 1D convolutional layer [Default: 0]
            device (torch.device, optional): Device for the tensors
        """
        super(CoverageAttention, self).__init__()
        
        self.alpha = None
        self.input_size = input_size
        self.output_size = output_size
        self.device = device

        self.conv_Q = nn.Conv2d(1, output_size, kernel_size=kernel_size, padding=padding) # same
        self.fc_Wa = nn.Linear(n, n_prime)
        self.conv_Ua = nn.Conv2d(input_size, n_prime, kernel_size=1)
        self.fc_Uf = nn.Linear(output_size, n_prime)
        self.fc_Va = nn.Linear(n_prime, 1)
        
        # init
        nn.init.xavier_normal_(self.fc_Wa.weight)
        nn.init.xavier_normal_(self.fc_Va.weight)
        nn.init.xavier_normal_(self.fc_Uf.weight)

    def reset_alpha(self, bs, attn_size): 
        """
        Args:
            attn_size: H*W (feature map size)
        """
        self.alpha = torch.zeros((bs, 1, attn_size)).cuda()

    def forward(self, x, st_hat): 
        bs = x.size(0) # x (bs, c, h, w)

        if self.alpha is None:
            self.reset_alpha(bs, x.size(2)*x.size(3))
            
        beta = self.alpha.sum(1)
        beta = beta.view(bs, x.size(2), x.size(3)) # 当前时间步之前的alpha累加 (bs, attn_size)

        F = self.conv_Q(beta.unsqueeze(1)) # (bs, output_size, h, w)
        F = F.permute(2, 3, 0, 1) # (h, w, bs, output_size)
        cover = self.fc_Uf(F) # (h, w, bs, n_prime)
        key = self.conv_Ua(x).permute(2, 3, 0, 1) # (h, w, bs, n_prime)
        query = self.fc_Wa(st_hat).squeeze(1) #(bs, n_prime)
          
        attention_score = torch.tanh(key + query[None, None, :, :] + cover)

        e_t = self.fc_Va(attention_score).squeeze(3) # (h, w, bs)
        e_t = e_t.permute(2, 0, 1).view(bs, -1) # (bs, h*w)
        e_t_exp = torch.exp(e_t)
        e_t_sum = e_t_exp.sum(1)
        alpha_t = torch.zeros((bs, x.size(2)*x.size(3)), device=self.device) # (bs, attn_size)
        for i in range(bs):
            e_t_div = e_t_exp[i]/(e_t_sum[i]+1e-8)
            alpha_t[i] = e_t_div
        self.alpha = torch.cat((self.alpha, alpha_t.unsqueeze(1)), dim=1)
        gt = alpha_t * x.view(bs, x.size(1), -1).transpose(0, 1) # x(bs, c, attn_size)->x(c, bs, attn_size)
        return gt.transpose(0, 1).sum(2), alpha_t.view(bs, 1, x.size(2), x.size(3))
    
class Maxout(nn.Module):
    """
    Maxout makes pools from the last dimension and keeps only the maximum value from
    each pool.
    """

    def __init__(self, pool_size):
        """
        Args:
            pool_size (int): Number of elements per pool
        """
        super(Maxout, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        [*shape, last] = x.size()
        out = x.view(*shape, last // self.pool_size, self.pool_size)
        out, _ = out.max(-1)
        return out

class Decoder(nn.Module):
    """Decoder
    GRU based Decoder which attends to the low- and high-resolution annotations to
    create a LaTeX string.
    """

    def __init__(
        self,
        num_classes,
        hidden_size=256,
        embedding_dim=256,
        checkpoint=None,
        device=device,
    ):
        """
        Args:
            num_classes (int): Number of symbol classes
            low_res_shape ((int, int, int)): Shape of the low resolution annotations
                i.e. (C, W, H)
            high_res_shape ((int, int, int)): Shape of the high resolution annotations
                i.e. (C_prime, 2W, 2H)
            hidden_size (int, optional): Hidden size of the GRU [Default: 256]
            embedding_dim (int, optional): Dimension of the embedding [Default: 256]
            checkpoint (dict, optional): State dictionary to be loaded
            device (torch.device, optional): Device for the tensors
        """
        super(Decoder, self).__init__()

        context_size = 684 
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.gru1 = nn.GRU(
            input_size=embedding_dim, hidden_size=hidden_size, batch_first=True
        )
        self.gru2 = nn.GRU(
            input_size=context_size, hidden_size=hidden_size, batch_first=True
        )
        self.coverage_attn = CoverageAttention(
            context_size,
            decoder_conv_filters,
            kernel_size=(11, 11),
            padding=5,
            device=device,
        )
        self.fc_Wo = nn.Linear(embedding_dim//2, num_classes)
        self.fc_Ws = nn.Linear(hidden_size, embedding_dim)
        self.fc_Wc = nn.Linear(context_size, embedding_dim)

        self.maxout = Maxout(2)
        self.hidden_size = hidden_size
        nn.init.xavier_normal_(self.fc_Wo.weight)
        nn.init.xavier_normal_(self.fc_Ws.weight)
        nn.init.xavier_normal_(self.fc_Wc.weight)

    def init_hidden(self, batch_size):
        return torch.zeros((1, batch_size, self.hidden_size))

    def reset(self, batch_size, feat_shape):
        self.coverage_attn.reset_alpha(batch_size, feat_shape[2]*feat_shape[3])
    
    # Unsqueeze and squeeze are used to add and remove the seq_len dimension,
    # which is always 1 since only the previous symbol is provided, not a sequence.
    # The inputs that are multiplied by the weights are transposed to get
    # (m x batch_size) instead of (batch_size x m). The result of the
    # multiplication is tranposed back.
    def forward(self, x, hidden, feature):
        embedded = self.embedding(x)
        pred, _ = self.gru1(embedded, hidden)
        # u_pred is computed here instead of in the coverage attention, because the
        # weight U_pred is shared and the coverage attention does not use pred for
        # anything else. This avoids computing it twice.
        context, decoder_attention = self.coverage_attn(feature, pred)
        new_hidden, _ = self.gru2(context.unsqueeze(1), pred.transpose(0, 1))
        
        w_s = self.fc_Ws(new_hidden.squeeze(1))
        w_c = self.fc_Wc(context)
        out = embedded.squeeze(1) + w_s + w_c
        out = self.maxout(out)
        out = self.fc_Wo(out)
        return out, new_hidden.transpose(0, 1), decoder_attention  
 
