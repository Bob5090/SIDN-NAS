import os

from torch.nn import GroupNorm, Module, Sequential, Conv2d, BatchNorm2d, ConvTranspose2d, ReLU, MaxPool2d, Sigmoid, Parameter, SELU
from torch import tensor, cat
import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial

from functools import partial

nonlinearity = partial(F.relu, inplace=True)


# activation funciton Mish, 
# class Mish(torch.nn.Module):
#     def forward(self, x):
#         return x * torch.tanh(torch.nn.functional.softplus(x))
#
#
# #activation function Swish
# class Swish(nn.Module):
#     def forward(self, x):
#         return x * torch.sigmoid(x)

    
# class DualConv(nn.Module):
#
#     def __init__(self, in_channels, out_channels, stride, g, k_size = (3, 3), is_bias = True, padding = (1, 1), padding_mode = 'zeros'):
#         """
#         Initialize the DualConv class.
#         :param input_channels: the number of input channels
#         :param output_channels: the number of output channels
#         :param stride: convolution stride
#         :param g: the value of G used in DualConv
#         """
#         super(DualConv, self).__init__()
#         # Group Convolution
#         self.gc = nn.Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=padding, groups=g, bias=is_bias, padding_mode = padding_mode)
#         # Pointwise Convolution
#         self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride, bias=is_bias, padding = padding, padding_mode = padding_mode)
#
#     def forward(self, input_data):
#         """
#         Define how DualConv processes the input images or input feature maps.
#         :param input_data: input images or input feature maps
#         :return: return output feature maps
#         """
#         return self.gc(input_data) + self.pwc(input_data)
    
class Conv2d_BN(torch.nn.Module):
    '''
    2d convolutional layers
    argument:
        in_channel {int} -- number of input filters
        out_channel {int} -- number of output filters
        kernel_size {tuple} -- size of the conv kernel
        stride {tuple} -- stride of the convolution (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})
    '''
    def __init__(self, in_channel, out_channel, kernel_size, stride = (1, 1),
                 activation = 'relu', is_bias = True, padding = (1, 1), padding_mode='zeros'):
        super().__init__()
        self.activation = activation
        self.conv1 = torch.nn.Conv2d(in_channels=in_channel,
                                     out_channels=out_channel,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     padding_mode=padding_mode)
#         self.conv1 = DualConv(in_channel, out_channel, stride = stride, g = 8, k_size = kernel_size, is_bias = is_bias, padding_mode=padding_mode)
        self.BN = torch.nn.BatchNorm2d(out_channel)
        if activation == 'swish':
            self.activation_fn = Swish()
        elif activation == 'mish':
            self.activation_fn = Mish()
        elif activation == 'selu':
            self.activation_fn = torch.nn.functional.selu
        else:
            self.activation_fn = torch.nn.functional.relu

    def forward(self, x):
        x = self.conv1(x)
        x = self.BN(x)
        return self.activation_fn(x)


# # Depth-wise separable conv
# class DWSC_2d(torch.nn.Module):
#     def __init__(self, in_channel, out_channel, kernel_size, stride=(1, 1), padding=(1, 1), activation='relu', is_bias=True):
#         super().__init__()
#         self.activation = activation
#         self.depth_wise = torch.nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channel, bias=is_bias)
#         self.point_wise = torch.nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, bias=is_bias)
#         self.BN = torch.nn.BatchNorm2d(out_channel)

#     def forward(self, x):
#         x = self.depth_wise(x)
#         x = self.point_wise(x)
#         x = self.BN(x)
#         return torch.nn.functional.relu(x) if self.activation == 'relu' else x

# attention 

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

    
class ResConv(torch.nn.Module):

    def __init__(self, in_channel, out_channel, BN_momentum=0.01, is_bias=True, is_inplace=True, activation='relu'):
        super().__init__()
        self.is_inplace = is_inplace
        self.in_channel = in_channel
        self.out_channel = out_channel
        conv_3x3_num = out_channel
        skip_num = out_channel

        if activation == 'swish':
            self.activation_fn = Swish()
        elif activation == 'mish':
            self.activation_fn = Mish()
        elif activation == 'selu':
            self.activation_fn = SELU()
        else:
            self.activation_fn = ReLU(inplace=True)

        self.skip_cut_1 = Conv2d_BN(in_channel, skip_num, kernel_size=(1, 1), activation='None', is_bias=is_bias, padding=0)
        self.skip_cut_2 = Conv2d_BN(in_channel, skip_num, kernel_size=(1, 1), activation='None', is_bias = is_bias, padding=0)

        self.conv_3x3_1 = Conv2d_BN(in_channel, conv_3x3_num, kernel_size=(3, 3), activation=activation, is_bias=is_bias)
        self.conv_3x3_2 = Conv2d_BN(in_channel, conv_3x3_num, kernel_size=(3, 3), activation=activation, is_bias=is_bias)

        self.batch_norm_1 = torch.nn.BatchNorm2d(out_channel, momentum=BN_momentum)
        self.batch_norm_2 = torch.nn.BatchNorm2d(out_channel, momentum=BN_momentum)
        self.acti_1 = self.activation_fn
        self.acti_2 = self.activation_fn

    def forward(self, x):
        # print(f'shape of input: ', x.shape)

        # print(f'skip_cut shape is', skip_cut.shape)
        skip_cut = self.skip_cut_1(x)
        x = self.conv_3x3_1(x)
        x = x + skip_cut
        x = self.batch_norm_1(x)
        x = self.acti_1(x)

        skip_cut = self.skp_cut_2(x)
        x = self.conv_3x3_2(x)
        x = x + skip_cut
        x = self.batch_norm_2(x)
        x = self.acti_2(x)
        return x

    
    
class Block(torch.nn.Module):
    
    def __init__(self, in_channel, out_channel, conv=None, batch_norm_momentum=0.01, last_decoder=False, acti = 'relu', layers=4):
        super(Block, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.layers = layers
        if acti == 'swish':
            self.activation_fn = Swish()
        elif acti == 'mish':
            self.activation_fn = Mish()
        elif acti == 'selu':
            self.activation_fn = SELU()
        else:
            self.activation_fn = ReLU(inplace=True)
        
        self.operator1 = ResConv(in_channel, out_channel, batch_norm_momentum, activation = acti) if conv is None else conv[0]
        self.operator2 = ResConv(in_channel, out_channel, batch_norm_momentum, activation = acti) if conv is None else conv[1]

        self.last_decoder = last_decoder

        # selection matrix
        alpha = torch.nn.Parameter(torch.ones((self.layers + 1, self.layers)), requires_grad=True)
        # alpha = torch.nn.Parameter(torch.ones((4, 3)), requires_grad=True)

        self.register_parameter('alpha', alpha)
        torch.nn.init.xavier_uniform_(self.alpha)

    def forward(self, x, skip):
        bn, c, w, h = x.size()
        device = x.device
        
        if skip is not None:
            aligned_skip = [F.interpolate(v, size=x.size()[-2:], mode='bilinear').unsqueeze(-1) for v in skip+[x]]
            aligned_skip = torch.cat(aligned_skip, dim=-1) # bn c w h 5

            score = F.gumbel_softmax(self.alpha, dim=0)
            
            # score = torch.eye(score.shape[0], score.shape[1], dtype=int).to(device)
            # print('choose:', score)
            skip = torch.matmul(aligned_skip, score) # bn c w h 3
            # print('skip.shape:', skip.shape)
            # weight matrix
            maxidx = torch.argmax(self.alpha, dim=0, keepdim=False)
            # maxidx = torch.tensor([0, 1, 2, 3, 4]).to(device)
            # print('maxidx:', maxidx)
            un, idx, counts = torch.unique(maxidx, return_inverse=True, return_counts=True, dim=0)
            # 全连接，
            weights = torch.ones(self.layers,).cuda(device) #cuda
            # weights = torch.ones(3,) #cpu

            # for i in range(idx.size(0)):
            #     weights[i] /= counts[idx[i]]

            # print('weights:', weights)
            # print('weight.view:', weights.view(1, 1, 1, 1, -1))
            
            skip = torch.sum(skip * weights.view(1, 1, 1, 1, -1), dim=-1, keepdim=False) # bn c w h 
            skip = skip + x
            
            x = skip/(un.size(0) + 1.)
     

        x = self.operator1(x)
        x = self.operator2(x)

        return x
 
    
    
class ResSkip(torch.nn.Module):

    def __init__(self, in_channel, out_channel, skip_len, isATT):
        super().__init__()
        self.length = skip_len
        self.short_cuts = torch.nn.ModuleList([])
        self.convs = torch.nn.ModuleList([])
        self.bns = torch.nn.ModuleList([])
        
        self.CSATT = CSATT(out_channel) if isATT else None

        if self.length > 0:
            for i in range(self.length):
                if i == 0:
                    self.short_cuts.append(Conv2d_BN(in_channel, out_channel, kernel_size=(1, 1), activation='None', padding=(0, 0)))
                    self.convs.append(Conv2d_BN(in_channel, out_channel, kernel_size=(3, 3), activation='relu'))
                else:
                    self.short_cuts.append(Conv2d_BN(out_channel, out_channel, kernel_size=(1, 1), activation='None', padding=(0, 0)))
                    self.convs.append(Conv2d_BN(out_channel, out_channel, kernel_size=(3, 3), activation='relu'))

                self.bns.append(torch.nn.BatchNorm2d(out_channel))
            # pass
        else:
            for i in range(-self.length):
                self.convs.append(Conv2d_BN(in_channel, out_channel, kernel_size=(3, 3), activation='relu'))


    def forward(self, x):
        if self.length > 0:
            for i in range(self.length):
                short_cut = self.short_cuts[i](x)
            
                x = self.convs[i](x)
            
                x = self.CSATT(x) + x  
            
                x = torch.nn.functional.relu(x, inplace=True)               
            
                x = self.bns[i](x)
                x = x + short_cut
            
                x = torch.nn.functional.relu(x, inplace=True)
            
                if i < self.length - 1:
                    x = F.max_pool2d(x, 2)
            
            return x

        else:
            for i in range(-self.length):
                x = self.convs[i](x)
                x = self.CSATT(x) + x
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        return x   
    

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CSATT(nn.Module):
    def __init__(self, channel):
        super(CSATT, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out

class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.interpolate(self.conv(self.pool1(x)), size=(h, w), mode='bilinear', align_corners=False)
        self.layer2 = F.interpolate(self.conv(self.pool2(x)), size=(h, w), mode='bilinear', align_corners=False)
        self.layer3 = F.interpolate(self.conv(self.pool3(x)), size=(h, w), mode='bilinear', align_corners=False)
        self.layer4 = F.interpolate(self.conv(self.pool4(x)), size=(h, w), mode='bilinear', align_corners=False)


        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out    
    
nonlinearity = partial(F.relu, inplace=True)
    
class SIDN(Module):

    def __init__(self,
                 in_channel: int = 3,
                 num_classes: int = 1,
                 iterations: int = 3,
                 multiplier: int = 2,
                 num_layers = 5,
                 acti = 'relu',
                 device_id = '0',
                 is_newBlock = True,
                 is_newSkip = True,
                 is_skipATT = True):
        super(SIDN, self).__init__()
        self.iterations = iterations
        self.multiplier = multiplier
        self.num_layers = num_layers
        self.batch_norm_momentum = 0.99
        self.initial_filter = 16 * self.multiplier
        self.activa_fn = acti
        # self.selu = torch.nn.functional.selu
        
        # self.device
        # global device
        # device = torch.device("cuda:" + device_id if torch.cuda.is_available() else "cpu")
        # define filters 
        self.filters_list = [16 * (self.multiplier + 1) for _ in range(self.num_layers + 1)]
            
        # self.dacblock = DACblock(self.filters_list[-1])
        # self.sppblock = SPPblock(self.filters_list[-1])
            
        print('Filters list is', self.filters_list)
        # 生成f[1]*512*512
        # preprocessing block
        self.pre_transform_conv_block = Sequential(
            Conv2d(in_channel, self.initial_filter, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            BatchNorm2d(self.initial_filter, momentum=self.batch_norm_momentum),
            ReLU(inplace=True),
            
            Conv2d(self.initial_filter, self.initial_filter, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            BatchNorm2d(self.initial_filter, momentum=self.batch_norm_momentum),
            ReLU(inplace=True),
            
            Conv2d(self.initial_filter, self.filters_list[0], kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            BatchNorm2d(self.filters_list[0], momentum=self.batch_norm_momentum),
            ReLU(inplace=True),

#             ResConv(in_channel, self.initial_filter, BN_momentum=self.batch_norm_momentum),
#             ResConv(self.initial_filter, self.initial_filter, BN_momentum=self.batch_norm_momentum),
#             ResConv(self.initial_filter, self.filters_list[0], BN_momentum=self.batch_norm_momentum),

            MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        )
      
        self.reuse_convs = []  
        self.encoders = nn.ModuleList()  
        self.reuse_deconvs = [] 
        self.decoders = nn.ModuleList()
        
        self.CSATTs = nn.ModuleList()
        
        enc_convs = []
        dec_convs = []
        
        # a 2-d array skip_path for all encoders
        self.encoder_skip_paths = nn.ModuleList()
        
        for iteration in range(self.iterations):
            for layer in range(self.num_layers):
                # encoder blocks
                in_channel = self.filters_list[layer]
                out_channel = self.filters_list[layer + 1]
                
                self.CSATTs.append(CSATT(out_channel))
                
                en_i_sk_path = nn.ModuleList()
                for i in range(self.num_layers, 0, -1):
                    length = i - layer
                    en_i_sk_path.append(ResSkip(in_channel, out_channel, skip_len=length if length >= 0 else length - 1, isATT=is_skipATT))

                self.encoder_skip_paths.append(en_i_sk_path)
                
                if iteration == 0:
                    conv1 = Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False)
                    conv2 = Conv2d(out_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False)
                    enc_convs.append((conv1, conv2))
#                     conv1 = ResConv(in_channel, out_channel, BN_momentum=self.batch_norm_momentum, is_bias=False)
#                     conv2 = ResConv(out_channel, out_channel, BN_momentum=self.batch_norm_momentum, is_bias=False)
#                     enc_convs.append((conv1, conv2))
                conv1 = enc_convs[layer][0]
                conv2 = enc_convs[layer][1]
                block = Block(in_channel, out_channel, (conv1, conv2), self.batch_norm_momentum, acti = self.activa_fn, layers = self.num_layers)

                self.add_module("iteration{0}_layer{1}_encoder_blocks".format(iteration, layer), block)
                # print("iteration{0}_layer{1}_encoder_blocks".format(iteration, layer),
                #       id("iteration{0}_layer{1}_encoder_blocks".format(iteration, layer)), id(conv1), id(conv2))
                self.encoders.append(block)

                # decoder blocks
                in_channel = self.filters_list[self.num_layers - layer]
                out_channel = self.filters_list[self.num_layers - 1 - layer]
                if iteration == 0:
                    conv1 = Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False)
                    conv2 = Conv2d(out_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False)
                    dec_convs.append((conv1, conv2))
#                     conv1 = ResConv(in_channel, out_channel, BN_momentum=self.batch_norm_momentum, is_bias=False)
#                     conv2 = ResConv(out_channel, out_channel, BN_momentum=self.batch_norm_momentum, is_bias=False)
#                     dec_convs.append((conv1, conv2))
                conv1 = dec_convs[layer][0]
                conv2 = dec_convs[layer][1]
        
                block = Block(in_channel, out_channel, (conv1, conv2), self.batch_norm_momentum,
                              last_decoder=iteration==self.iterations-1, acti = self.activa_fn, layers = self.num_layers)
            
            
                self.add_module("iteration{0}_layer{1}_decoder_blocks".format(iteration, layer), block)
                # print("iteration{0}_layer{1}_decoder_blocks".format(iteration, layer),
                #       id("iteration{0}_layer{1}_decoder_blocks".format(iteration, layer)), id(conv1), id(conv2))
                
                self.decoders.append(block)

                
#         print('The skip paths are:')
#         print(self.encoder_skip_paths)
#         os.system('pause')
        # bridge block
        self.middles = Sequential(
            Conv2d(self.filters_list[-1], self.filters_list[-1], kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            BatchNorm2d(self.filters_list[-1], momentum=self.batch_norm_momentum),
            ReLU(),
            
            Conv2d(self.filters_list[-1], self.filters_list[-1], kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            BatchNorm2d(self.filters_list[-1], momentum=self.batch_norm_momentum),
            ReLU(),
#             ResConv(self.filters_list[-1], self.filters_list[-1], is_bias=False, is_inplace=False),
#             ResConv(self.filters_list[-1], self.filters_list[-1], is_bias=False, is_inplace=False)
        )
        
        # self.bridge = Sequential(
        #     Conv2d(self.filters_list[-1] + 4, self.filters_list[-1], kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
        #     BatchNorm2d(self.filters_list[-1], momentum=self.batch_norm_momentum),
        #     ReLU(),
        # )
        # postprocessing block
        self.post_transform_conv_block = Sequential(
            Conv2d(self.filters_list[0], self.initial_filter, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            BatchNorm2d(self.initial_filter, momentum=self.batch_norm_momentum),
            ReLU(inplace=True),
            
            Conv2d(self.initial_filter, self.initial_filter, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            BatchNorm2d(self.initial_filter, momentum=self.batch_norm_momentum),
            ReLU(inplace=True),
            
            Conv2d(self.initial_filter, num_classes, kernel_size=(1, 1), stride=(1, 1), bias=False),
            Sigmoid()
#             ResConv(self.filters_list[0], self.initial_filter, BN_momentum=self.batch_norm_momentum, is_bias=False),
#             ResConv(self.initial_filter, self.initial_filter, BN_momentum=self.batch_norm_momentum, is_bias=False),
#             Conv2d(self.initial_filter, num_classes, kernel_size=(1, 1), stride=(1, 1), bias=False),
#             Sigmoid()
        )

    def forward(self, x):
        enc = [None for _ in range(self.num_layers)]
        dec = [None for _ in range(self.num_layers)]
        
        enc_skip_outputs = [None for _ in range(self.num_layers)]
        
        enc = [x] + enc

        x = self.pre_transform_conv_block(x)
        e_i = 0
        d_i = 0
        for iteration in range(self.iterations):
            # encoding path
            for layer in range(self.num_layers):
                if layer == 0:
                    x_in = x

#                 x_in = self.encoders[e_i](x_in, dec if iteration != 0 else None)
                
                en_i_skip_out = []
                x_in = self.encoders[e_i](x_in, dec if iteration != 0 else None)

                # new skip-connect
                for path in self.encoder_skip_paths[layer]:
                    en_i_skip_out.append(path(x_in))
                enc_skip_outputs[layer] = en_i_skip_out

                # CSATT
                x_in = self.CSATTs[e_i](x_in) + x_in
                
                enc[layer+1] = x_in
                x_in = F.max_pool2d(x_in, 2)
#                 x_in = F.
                e_i = e_i + 1

            # bridging
            x_in = self.middles(x_in)
            
#             print(x_in.shape)
#             x_in = self.dacblock(x_in)
# #             print(x_in.shape)
#             x_in = self.sppblock(x_in)
# #             print(x_in.shape)
            
#             x_in = self.bridge(x_in)
            
            x_in = F.interpolate(x_in, size=enc[-1].size()[-2:], mode='bilinear', align_corners=True)
            
            # skip-connects,transpose
            enc_skip_outputs = [list(item) for item in zip(*enc_skip_outputs)]

            # decodong path
            for layer in range(self.num_layers):
#                 x_in = self.decoders[d_i](x_in, enc[1:])
                x_in = self.decoders[d_i](x_in, enc_skip_outputs[layer])
                dec[layer] = x_in
                x_in = F.interpolate(x_in, size=enc[-1 - layer - 1].size()[-2:], mode='bilinear', align_corners=True)
                d_i = d_i + 1

        x_in = self.post_transform_conv_block(x_in)
        return x_in
