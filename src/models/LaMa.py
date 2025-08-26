import numpy as np

from .ffc import *
from .layers import *
from .vmamba import *
from .mambaIR import *

class ResnetBlock_remove_IN(nn.Module):
    def __init__(self, dim, dilation=1):
        super(ResnetBlock_remove_IN, self).__init__()

        self.ffc1 = FFC_BN_ACT(dim, dim, 3, 0.75, 0.75, stride=1, padding=1, dilation=dilation, groups=1, bias=False,
                               norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, enable_lfu=False)

        self.ffc2 = FFC_BN_ACT(dim, dim, 3, 0.75, 0.75, stride=1, padding=1, dilation=1, groups=1, bias=False,
                               norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, enable_lfu=False)

    def forward(self, x):
        output = x
        _, c, _, _ = output.shape
        output = torch.split(output, [c - int(c * 0.75), int(c * 0.75)], dim=1)
        x_l, x_g = self.ffc1(output)
        output = self.ffc2((x_l, x_g))
        output = torch.cat(output, dim=1)
        output = x + output

        return output


class MaskedSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids):
        """`input_ids` is expected to be [bsz x seqlen]."""
        return super().forward(input_ids)


class MultiLabelEmbedding(nn.Module):
    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_positions, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, input_ids):
        # input_ids:[B,HW,4](onehot)
        out = torch.matmul(input_ids, self.weight)  # [B,HW,dim]
        return out


class LaMa_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.pad0 = nn.ReflectionPad2d(3)
        self.conv0 = nn.Conv2d(in_channels=4, out_channels=48, kernel_size=7, padding=0)
        self.bn0 = nn.BatchNorm2d(48)
        self.act = nn.ReLU(True)
        
        self.conv1 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(48)
        
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(96)

        self.conv3 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(192)

        self.conv4 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(384)
        
        #++++++++++++++++++++
        self.linear1 = nn.Linear(64, 128).cuda()
        self.linear2 = nn.Linear(128, 256).cuda()
        self.linear3 = nn.Linear(256, 512).cuda()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.upl = []
        self.upl.append(nn.Linear(512, 256).cuda())
        self.upl.append(nn.Linear(256, 128).cuda())
        self.upl.append(nn.Linear(128, 64).cuda())
        
        #-------------------
        blocks = []
        ### resnet blocks
        self.norm_mid = nn.LayerNorm(384).cuda()
        self.norm1 = nn.LayerNorm(48).cuda()
        self.norm2 = nn.LayerNorm(48).cuda()
        self.norm3 = nn.LayerNorm(96).cuda()
        self.norm4 = nn.LayerNorm(192).cuda()
        
        self.patch_embed = PatchEmbed(
            img_size=32,
            patch_size=8,
            in_chans=384,
            embed_dim=384,
            norm_layer=nn.LayerNorm)
        
        self.patch_unembed = PatchUnEmbed(
            img_size=32,
            patch_size=8,
            in_chans=384,
            embed_dim=384,
            norm_layer=nn.LayerNorm)
        
        '''
        self.upsample_mode = UpsampleMode(UpsampleMode.NONTRAINABLE)
        self.blocks_up = (1, 1, 1)
        self.spatial_dims = 2
        self.init_filters = 64
        self.norm = ("GROUP", {"num_groups": 8})
        self.up_layers, self.up_samples = self._make_up_layers()
        '''
        #++++++++++++++++
        
        
        self.patch_embed1 = PatchEmbed(
            img_size=256,
            patch_size=8,
            in_chans=48,
            embed_dim=48,
            norm_layer=nn.LayerNorm)
        
        self.patch_unembed1 = PatchUnEmbed(
            img_size=256,
            patch_size=8,
            in_chans=48,
            embed_dim=48,
            norm_layer=nn.LayerNorm)
        
        self.patch_embed2 = PatchEmbed(
            img_size=128,
            patch_size=8,
            in_chans=48,
            embed_dim=48,
            norm_layer=nn.LayerNorm)
        
        self.patch_unembed2 = PatchUnEmbed(
            img_size=128,
            patch_size=8,
            in_chans=48,
            embed_dim=48,
            norm_layer=nn.LayerNorm)
        
        self.patch_embed3 = PatchEmbed(
            img_size=64,
            patch_size=8,
            in_chans=96,
            embed_dim=96,
            norm_layer=nn.LayerNorm)
        
        self.patch_unembed3 = PatchUnEmbed(
            img_size=64,
            patch_size=8,
            in_chans=96,
            embed_dim=96,
            norm_layer=nn.LayerNorm)
        
        self.patch_embed4 = PatchEmbed(
            img_size=64,
            patch_size=8,
            in_chans=192,
            embed_dim=192,
            norm_layer=nn.LayerNorm)
        
        self.patch_unembed4 = PatchUnEmbed(
            img_size=64,
            patch_size=8,
            in_chans=192,
            embed_dim=192,
            norm_layer=nn.LayerNorm)
        
        

        
        
        self.down1 = ResidualGroup(
                dim=48,
                input_resolution=(256, 256),
                depth=2,
                mlp_ratio=2,
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=True,
                img_size=256,
                patch_size=8,
                resi_connection='1conv')
        
        self.down2 = ResidualGroup(
                dim=48,
                input_resolution=(128, 128),
                depth=2,
                mlp_ratio=2,
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=True,
                img_size=128,
                patch_size=8,
                resi_connection='1conv')
        
        
        
        self.down3 = ResidualGroup(
                dim=96,
                input_resolution=(64, 64),
                depth=2,
                mlp_ratio=2,
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=True,
                img_size=64,
                patch_size=8,
                resi_connection='1conv')
                
        
        self.down4 = ResidualGroup(
                dim=192,
                input_resolution=(64, 64),
                depth=8,
                mlp_ratio=2,
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=True,
                img_size=64,
                patch_size=8,
                resi_connection='1conv')
                
        
        #-----------------
        
        embed_dim = 384
        for i in range(3):
            cur_resblock = ResidualGroup(
                dim=embed_dim,
                input_resolution=(64, 64),
                depth=6,
                mlp_ratio=2,
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=True,
                img_size=64,
                patch_size=8,
                resi_connection='1conv')
            blocks.append(cur_resblock)

        self.middle = nn.Sequential(*blocks)

        self.convt1 = nn.ConvTranspose2d(384, 192, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bnt1 = nn.BatchNorm2d(192)

        self.convt2 = nn.ConvTranspose2d(192, 96, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bnt2 = nn.BatchNorm2d(96)

        self.convt3 = nn.ConvTranspose2d(96, 48, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bnt3 = nn.BatchNorm2d(48)
        
        #self.convt4 = nn.ConvTranspose2d(48, 48, kernel_size=3, stride=2, padding=1, output_padding=1)
        #self.bnt4 = nn.BatchNorm2d(48)

        self.padt = nn.ReflectionPad2d(3)
        self.convt5 = nn.Conv2d(in_channels=48, out_channels=3, kernel_size=7, padding=0)
        self.act_last = nn.Tanh()

        self.side1 = nn.Conv2d(in_channels=384, out_channels=48, kernel_size=1, stride=1, padding=0)
        
        self.side2 = nn.Conv2d(in_channels=192, out_channels=48, kernel_size=1, stride=1, padding=0)
        
        self.side3 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=1, stride=1, padding=0)
        
        self.side4 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=1, stride=1, padding=0)
        
        self.fuse = nn.Conv2d(in_channels=6 * 48, out_channels=48, kernel_size=1, stride=1, padding=0)
        
    
    def forward(self, x, rel_pos_emb=None, direct_emb=None, str_feats=None):
        #down_x = []        
        
        x = self.pad0(x)
        x = self.conv0(x)
        x = self.bn0(x.to(torch.float32))
        x = self.act(x)
        
        #down_x.append(x)
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed1(x)
        x = self.down1(x, x_size)
        x = self.norm1(x)  
        x = self.patch_unembed1(x, x_size)
        
        x = self.conv1(x)
        x = self.bn1(x.to(torch.float32))
        x = self.act(x)

        #down_x.append(x)        
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed2(x)
        x = self.down2(x, x_size)
        x = self.norm2(x)  
        x = self.patch_unembed2(x, x_size)
        
        x = self.conv2(x)
        x = self.bn2(x.to(torch.float32))
        x = self.act(x)

        #down_x.append(x)        
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed3(x)
        x = self.down3(x, x_size)
        x = self.norm3(x)  
        x = self.patch_unembed3(x, x_size)
        
        x = self.conv3(x)
        x = self.bn3(x.to(torch.float32))
        x = self.act(x)

        #down_x.append(x)
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed4(x)
        x = self.down4(x, x_size)
        x = self.norm4(x)  
        x = self.patch_unembed4(x, x_size)
        
        x = self.conv4(x)
        x = self.bn4(x.to(torch.float32))
        x = self.act(x)
        # middle-start
        
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        
        for layer in self.middle:
            x = layer(x, x_size)
        
        x = self.norm_mid(x)  
        x = self.patch_unembed(x, x_size)
        
        # middle-end
        #down_x.reverse()
       
        out1 = F.interpolate(self.side1(x), scale_factor=8, mode='bilinear')
        
        x = self.convt1(x)
        x = self.bnt1(x.to(torch.float32))
        x = self.act(x)
        
        out2 = F.interpolate(self.side2(x), scale_factor=4, mode='bilinear')
        
        x = self.convt2(x)
        x = self.bnt2(x.to(torch.float32))
        x = self.act(x)
        
        out3 = F.interpolate(self.side3(x), scale_factor=2, mode='bilinear')
        
        x = self.convt3(x)
        x = self.bnt3(x.to(torch.float32))
        x = self.act(x)
        
        out4 = self.side4(x)

        x = torch.cat([out4, out3, out2, out1],1)
        x = self.fuse(x)
              
        x = self.padt(x)
        x = self.convt5(x)
        x = self.act_last(x)
        x = (x + 1) / 2
        return x



class ReZeroFFC(LaMa_model):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x, rel_pos_emb=None, direct_emb=None, str_feats=None):
        x = self.pad0(x)
        x = self.conv0(x)
        x = self.bn0(x.to(torch.float32))
        x = self.act(x)
        
        #down_x.append(x)
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed1(x)
        x = self.down1(x, x_size)
        x = self.norm1(x)  
        x = self.patch_unembed1(x, x_size)
        
        x = self.conv1(x)
        x = self.bn1(x.to(torch.float32))
        x = self.act(x)

        #down_x.append(x)        
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed2(x)
        x = self.down2(x, x_size)
        x = self.norm2(x)  
        x = self.patch_unembed2(x, x_size)
        
        x = self.conv2(x)
        x = self.bn2(x.to(torch.float32))
        x = self.act(x)

        #down_x.append(x)        
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed3(x)
        x = self.down3(x, x_size)
        x = self.norm3(x)  
        x = self.patch_unembed3(x, x_size)
        
        x = self.conv3(x)
        x = self.bn3(x.to(torch.float32))
        x = self.act(x)

        #down_x.append(x)
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed4(x)
        x = self.down4(x, x_size)
        x = self.norm4(x)  
        x = self.patch_unembed4(x, x_size)
        
        x = self.conv4(x)
        x = self.bn4(x.to(torch.float32))
        x = self.act(x)
        # middle-start
        
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        
        for layer in self.middle:
            x = layer(x, x_size)
        
        x = self.norm_mid(x)  
        x = self.patch_unembed(x, x_size)
        
        # middle-end
        #down_x.reverse()
       
        out1 = F.interpolate(self.side1(x), scale_factor=8, mode='bilinear')
        
        x = self.convt1(x)
        x = self.bnt1(x.to(torch.float32))
        x = self.act(x)
        
        out2 = F.interpolate(self.side2(x), scale_factor=4, mode='bilinear')
        
        x = self.convt2(x)
        x = self.bnt2(x.to(torch.float32))
        x = self.act(x)
        
        out3 = F.interpolate(self.side3(x), scale_factor=2, mode='bilinear')
        
        x = self.convt3(x)
        x = self.bnt3(x.to(torch.float32))
        x = self.act(x)
        
        out4 = self.side4(x)

        x = torch.cat([out4, out3, out2, out1],1)
        x = self.fuse(x)
         
        x = self.padt(x)
        x = self.convt5(x)
        x = self.act_last(x)
        x = (x + 1) / 2
        return x



class StructureEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.rezero_for_mpe is None:
            self.rezero_for_mpe = False
        else:
            self.rezero_for_mpe = config.rezero_for_mpe

        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = GateConv(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = nn.ReLU(True)

        self.conv2 = GateConv(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        blocks = []
        # resnet blocks
        for i in range(3):
            blocks.append(ResnetBlock(input_dim=512, out_dim=None, dilation=2))

        self.middle = nn.Sequential(*blocks)
        self.alpha1 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        self.convt1 = GateConv(512, 256, kernel_size=4, stride=2, padding=1, transpose=True)
        self.bnt1 = nn.BatchNorm2d(256)
        self.alpha2 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        self.convt2 = GateConv(256, 128, kernel_size=4, stride=2, padding=1, transpose=True)
        self.bnt2 = nn.BatchNorm2d(128)
        self.alpha3 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        self.convt3 = GateConv(128, 64, kernel_size=4, stride=2, padding=1, transpose=True)
        self.bnt3 = nn.BatchNorm2d(64)
        self.alpha4 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        if self.rezero_for_mpe:
            self.rel_pos_emb = MaskedSinusoidalPositionalEmbedding(num_embeddings=config.rel_pos_num,
                                                                   embedding_dim=64)
            self.direct_emb = MultiLabelEmbedding(num_positions=4, embedding_dim=64)
            self.alpha5 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)
            self.alpha6 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

    def forward(self, x, rel_pos=None, direct=None):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x.to(torch.float32))
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x.to(torch.float32))
        x = self.act(x)

        x = self.conv3(x)
        x = self.bn3(x.to(torch.float32))
        x = self.act(x)

        x = self.conv4(x)
        x = self.bn4(x.to(torch.float32))
        x = self.act(x)

        return_feats = []
        x = self.middle(x)
        return_feats.append(x * self.alpha1)

        x = self.convt1(x)
        x = self.bnt1(x.to(torch.float32))
        x = self.act(x)
        return_feats.append(x * self.alpha2)

        x = self.convt2(x)
        x = self.bnt2(x.to(torch.float32))
        x = self.act(x)
        return_feats.append(x * self.alpha3)

        x = self.convt3(x)
        x = self.bnt3(x.to(torch.float32))
        x = self.act(x)
        return_feats.append(x * self.alpha4)

        return_feats = return_feats[::-1]

        if not self.rezero_for_mpe:
            return return_feats
        else:
            b, h, w = rel_pos.shape
            rel_pos = rel_pos.reshape(b, h * w)
            rel_pos_emb = self.rel_pos_emb(rel_pos).reshape(b, h, w, -1).permute(0, 3, 1, 2) * self.alpha5
            direct = direct.reshape(b, h * w, 4).to(torch.float32)
            direct_emb = self.direct_emb(direct).reshape(b, h, w, -1).permute(0, 3, 1, 2) * self.alpha6

            return return_feats, rel_pos_emb, direct_emb


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc):
        super().__init__()
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_nc, out_channels=64, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kw, stride=2, padding=padw)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=kw, stride=2, padding=padw)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=kw, stride=2, padding=padw)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=kw, stride=1, padding=padw)
        self.bn5 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 1, kernel_size=kw, stride=1, padding=padw)

    def forward(self, x):
        conv1 = self.conv1(x)

        conv2 = self.conv2(conv1)
        conv2 = self.bn2(conv2.to(torch.float32))
        conv2 = self.act(conv2)

        conv3 = self.conv3(conv2)
        conv3 = self.bn3(conv3.to(torch.float32))
        conv3 = self.act(conv3)

        conv4 = self.conv4(conv3)
        conv4 = self.bn4(conv4.to(torch.float32))
        conv4 = self.act(conv4)

        conv5 = self.conv5(conv4)
        conv5 = self.bn5(conv5.to(torch.float32))
        conv5 = self.act(conv5)

        conv6 = self.conv6(conv5)

        outputs = conv6

        return outputs, [conv1, conv2, conv3, conv4, conv5]
