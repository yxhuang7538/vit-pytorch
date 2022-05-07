# 创建vit模型
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    # 将2维图像转化为patch编码
    # 224x224x3-->((224/16)^2)x(16x16x3)-->196x768  
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_c=3, 
                 embed_dim=768, 
                 norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1]) # 14 x 14
        self.num_patches = self.grid_size[0] * self.grid_size[1] # 总共被分割为了多少个patch
        self.embed_dim = self.patch_size[0] * self.patch_size[1] # 编码长度 16x16

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # proj : [B, 3, 224, 224] -> [B, 16x16x3, 14, 14]
        # flatten: [B, 16x16x3, 14, 14] -> [B, 16x16x3, 14x14]
        # transpose: [B, 16x16x3, 14x14] -> [B, 14x14, 16x16x3]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

def _init_vit_weights(m):
    """
    ViT模型权重初始化
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

class Block(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_ratio=0.,
                 attn_drop_ratio=0., 
                 drop_out_ratio=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm):
        # 整体结构顺序
        # ->norm1->Multi-Head-Attention->Dropout->norm2->MLP->Dropout->
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        
        self.drop_out = nn.Dropout(p=drop_out_ratio)

        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MlP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_out(self.attn(self.norm1(x)))
        x = x + self.drop_out(self.mlp(self.norm2(x)))
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads # 分给多个头
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop_ratio)

        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: 矩阵乘法 -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: 矩阵乘法 -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MlP(nn.Module):
    """
    MLP结构
    """
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 out_features=None, 
                 act_layer=nn.GELU, 
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()

        self.fc2 = nn.Linear(hidden_features, out_features)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class VisionTransformer(nn.Module):
    # vit结构整体设计
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_c=3, 
                 num_classes=6, 
                 embed_dim=768, 
                 depth=12, 
                 num_heads=12, 
                 mlp_ratio=4.0, 
                 qkv_bias=True,
                 qk_scale=None, 
                 representation_size=None, 
                 drop_ratio=0,
                 attn_drop_ratio=0,  
                 embed_layer=PatchEmbed):
        '''
        参数说明：
        img_size : 输入图片大小
        patch_size : 分割的patch大小
        in_c : 输入通道数
        num_classes : 类别数量
        embed_dim : 编码维度
        depth : tf的深度
        num_heads : 注意力头的数量
        mlp_ratio : MLP模块中的膨胀系数
        qkv_bias : 计算qkv时考虑偏置bias
        qk_scale : 如果设置了将取代计算qk分数时分母位置的放缩系数
        representation_size : 如果设置则将启用并设置pre-logits层为此值（预先表征）
        drop_ratio : dropout 概率
        attn_drop_ratio : 注意力层的dropout概率
        embed_layer : patch的编码层，将patch映射到一维空间中
        '''
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.num_tokens = 1 # 需要多添加的一个类别token
        norm_layer = partial(nn.LayerNorm, eps=1e-6) # 若无特殊正则化层，则会选择nn.LayerNorm
        act_layer = nn.GELU # 若未传入激活层则用nn.GELU

        # 224x224x3-->((224/16)^2)x(16x16x3)-->196x768  
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, 
                                      in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches # patch的个数

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # 1x1x768
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim)) # 1x(14x14 + 1)x768
        self.pos_drop = nn.Dropout(p=drop_ratio)

        # depth个encoder进行堆叠
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
                  norm_layer=norm_layer, act_layer=act_layer)
            for _ in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        # 预表征层
        if representation_size:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # 分类头
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # 初始化权重
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) # [1, 1, 768] -> [B, 1, 768]
        x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.pre_logits(x[:, 0])

        x = self.head(x)
        return x


if __name__ == '__main__':
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=5)
    print(model)