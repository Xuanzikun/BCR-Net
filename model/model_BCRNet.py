import torch
import torch.nn as nn
import torch.nn.functional as F


def get_norm(norm, num_channels, num_groups):
    if norm == 'in':
        return nn.InstanceNorm2d(num_channels, affine=True)
    if norm == 'bn':
        return nn.BatchNorm2d(num_channels)
    if norm == 'gn':
        ng = min(num_groups, num_channels)
        while ng > 1 and num_channels % ng != 0:
            ng -= 1
        return nn.GroupNorm(ng, num_channels)
    if norm is None:
        return nn.Identity()
    raise ValueError('unknown normalization type')


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)

    def forward(self, x):
        if x.shape[2] % 2 == 1:
            raise ValueError('downsampling tensor height should be even')
        if x.shape[3] % 2 == 1:
            raise ValueError('downsampling tensor width should be even')
        return self.downsample(x)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )

    def forward(self, x):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, norm='gn', num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        self.norm = get_norm(norm, in_channels, num_groups)
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.to_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = torch.split(self.to_qkv(self.norm(x)), self.in_channels, dim=1)
        q = q.permute(0, 2, 3, 1).view(b, h * w, c)
        k = k.view(b, c, h * w)
        v = v.permute(0, 2, 3, 1).view(b, h * w, c)
        dot_products = torch.bmm(q, k) * (c ** (-0.5))
        attention = torch.softmax(dot_products, dim=-1)
        out = torch.bmm(attention, v)
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)
        return self.to_out(out) + x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dropout,
        activation=F.relu,
        norm='gn',
        num_groups=32,
        use_attention=False,
    ):
        super().__init__()
        self.activation = activation
        self.norm_1 = get_norm(norm, in_channels, num_groups)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm_2 = get_norm(norm, out_channels, num_groups)
        self.conv_2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.residual_connection = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.attention = nn.Identity() if not use_attention else AttentionBlock(out_channels, norm, num_groups)
        self.scale = nn.Parameter(torch.ones((1, out_channels, 1, 1)), requires_grad=True)

    def forward(self, x):
        out = self.activation(self.norm_1(x))
        out = self.conv_1(out)
        out = self.activation(self.norm_2(out))
        out = self.conv_2(out) * self.scale + self.residual_connection(x)
        out = self.attention(out)
        return out


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        norm='gn',
        num_groups=32,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            get_norm(norm, out_ch, num_groups),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MCEM(nn.Module):
    """Multi-scale Context Enhancement Module."""

    def __init__(self, channels, norm='gn', num_groups=32, dilations=(1, 2, 3)):
        super().__init__()
        if len(dilations) != 3:
            raise ValueError(f'MCEM expects exactly 3 dilation rates, got {dilations}')
        d1, d2, d3 = [int(d) for d in dilations]
        self.branch1 = ConvBNReLU(channels, channels, 3, padding=d1, dilation=d1, norm=norm, num_groups=num_groups)
        self.branch2 = ConvBNReLU(channels, channels, 3, padding=d2, dilation=d2, norm=norm, num_groups=num_groups)
        self.branch3 = ConvBNReLU(channels, channels, 3, padding=d3, dilation=d3, norm=norm, num_groups=num_groups)
        self.fuse = ConvBNReLU(channels * 3, channels, kernel_size=1, padding=0, norm=norm, num_groups=num_groups)

    def forward(self, x):
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f3 = self.branch3(x)
        return self.fuse(torch.cat([f1, f2, f3], dim=1))


class GateNet(nn.Module):
    def __init__(self, in_ch, out_ch, norm='gn', num_groups=32):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            get_norm(norm, out_ch, num_groups),
            nn.ReLU(inplace=True),
        )
        self.gate_out = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        nn.init.constant_(self.gate_out.bias, -2.0)

    def forward(self, x):
        return self.gate_out(self.pre(x))


def _red_ch(c, reduction):
    return max(1, c // max(1, reduction))


class BoundaryGuidedFusion(nn.Module):
    def __init__(self, body_channels, bound_channels, reduction=16):
        super().__init__()
        rb = _red_ch(bound_channels, reduction)
        rbody = _red_ch(body_channels, reduction)
        self.body_attn = nn.Sequential(
            nn.Conv2d(bound_channels, rb, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(rb, 1, 1),
            nn.Sigmoid(),
        )
        self.bound_attn = nn.Sequential(
            nn.Conv2d(body_channels, rbody, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(rbody, 1, 1),
            nn.Sigmoid(),
        )
        self.fuse_conv = nn.Conv2d(body_channels + bound_channels, body_channels, 1, bias=True)

    def forward(self, body_feat, bound_feat):
        attn_body = self.body_attn(bound_feat)
        body_enhanced = body_feat * attn_body
        attn_bound = self.bound_attn(body_feat)
        bound_enhanced = bound_feat * attn_bound
        body_out = body_feat + body_enhanced
        bound_out = bound_feat + bound_enhanced
        fused = self.fuse_conv(torch.cat([body_enhanced, bound_enhanced], dim=1))
        return body_out, bound_out, fused

class BCRNet(nn.Module):
    """
    Body-boundary collaborative network with adaptive residual compensation.

    The network contains a shared encoder, mutually guided body and boundary
    decoders, an MCEM-based compensation stream, and GARF refinement.

    Outputs:
      - pred_body
      - pred_bound
      - pred_aux
      - pred_final
    """

    def __init__(
        self,
        img_channels=1,
        base_channels=16,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        activation=F.relu,
        dropout=0.2,
        norm='gn',
        num_groups=16,
    ):
        super().__init__()
        self.activation = activation
        self.img_channels = img_channels
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.num_res_blocks = num_res_blocks
        self.norm = norm
        self.num_groups = num_groups

        self.init_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)
        self.second_conv = nn.Conv2d(base_channels, base_channels, 3, padding=1)

        self.downs = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels

        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(
                    ResidualBlock(
                        now_channels,
                        out_channels,
                        dropout,
                        activation=activation,
                        norm=norm,
                        num_groups=num_groups,
                        use_attention=False,
                    )
                )
                now_channels = out_channels
                channels.append(now_channels)
            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(now_channels))
                channels.append(now_channels)

        self.mid = nn.ModuleList(
            [
                ResidualBlock(
                    now_channels,
                    now_channels,
                    dropout,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=True,
                ),
                ResidualBlock(
                    now_channels,
                    now_channels,
                    dropout,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=False,
                ),
            ]
        )

        ch_stack = list(channels)
        now_b = now_channels
        now_bd = max(8, now_channels // 2)
        self.bound_in = nn.Conv2d(now_channels, now_bd, 1, bias=False)

        self.body_res = nn.ModuleList()
        self.bound_res = nn.ModuleList()
        self.bgfm = nn.ModuleList()
        self.body_up = nn.ModuleList()
        self.bound_up = nn.ModuleList()

        for i, mult in reversed(list(enumerate(channel_mults))):
            out_b = base_channels * mult
            out_bd = max(8, out_b // 2)
            for _ in range(num_res_blocks + 1):
                skip_c = ch_stack.pop()
                self.body_res.append(
                    ResidualBlock(
                        skip_c + now_b,
                        out_b,
                        dropout,
                        activation=activation,
                        norm=norm,
                        num_groups=num_groups,
                        use_attention=False,
                    )
                )
                self.bound_res.append(
                    ResidualBlock(
                        skip_c + now_bd,
                        out_bd,
                        dropout,
                        activation=activation,
                        norm=norm,
                        num_groups=num_groups,
                        use_attention=False,
                    )
                )
                now_b, now_bd = out_b, out_bd
            self.bgfm.append(BoundaryGuidedFusion(now_b, now_bd))
            if i != 0:
                self.body_up.append(Upsample(now_b))
                self.bound_up.append(Upsample(now_bd))

        assert len(ch_stack) == 0

        mid_level = min(2, len(channel_mults) - 1)
        self.fm_channels = base_channels * channel_mults[mid_level]
        self.aux_joint = ConvBNReLU(
            base_channels + self.fm_channels,
            base_channels,
            3,
            padding=1,
            norm=norm,
            num_groups=num_groups,
        )
        self.mcem = MCEM(base_channels, norm=norm, num_groups=num_groups)
        self.aux_proj = ConvBNReLU(
            base_channels,
            base_channels,
            3,
            padding=1,
            norm=norm,
            num_groups=num_groups,
        )

        self.gate_net = GateNet(
            base_channels * 3,
            base_channels,
            norm=norm,
            num_groups=num_groups,
        )

        self.head_body_norm = get_norm(norm, base_channels, num_groups)
        self.head_body = nn.Conv2d(base_channels, img_channels, 3, padding=1)

        self.head_bound_norm = get_norm(norm, now_bd, num_groups)
        self.head_bound = nn.Conv2d(now_bd, img_channels, 3, padding=1)

        self.aux_head = nn.Conv2d(base_channels, img_channels, kernel_size=1)

        self.head_final_norm = get_norm(norm, base_channels, num_groups)
        self.head_final = nn.Conv2d(base_channels, img_channels, 3, padding=1)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.second_conv(x)
        fs = x

        skips = [x]
        fm = None
        for layer in self.downs:
            x = layer(x)
            skips.append(x)
            if fm is None and x.shape[1] == self.fm_channels:
                fm = x

        if fm is None:
            fm = x

        for layer in self.mid:
            x = layer(x)

        body_feat = x
        bound_feat = self.bound_in(x)

        ri = 0
        fused_last = None
        num_levels = len(self.channel_mults)

        for level in range(num_levels):
            for _ in range(self.num_res_blocks + 1):
                sk = skips.pop()
                body_feat = self.body_res[ri](torch.cat([body_feat, sk], dim=1))
                bound_feat = self.bound_res[ri](torch.cat([bound_feat, sk], dim=1))
                ri += 1

            body_feat, bound_feat, fused_last = self.bgfm[level](body_feat, bound_feat)

            if level < num_levels - 1:
                body_feat = self.body_up[level](body_feat)
                bound_feat = self.bound_up[level](bound_feat)

        fm_up = F.interpolate(fm, size=fs.shape[-2:], mode='bilinear', align_corners=False)
        f_sm = self.aux_joint(torch.cat([fs, fm_up], dim=1))
        f_ctx = self.mcem(f_sm)
        f_aux = self.aux_proj(f_ctx)

        f_main = fused_last
        f_diff = f_aux - f_main
        joint = torch.cat([f_main, f_aux, torch.abs(f_diff)], dim=1)
        gate = torch.sigmoid(self.gate_net(joint))
        f_refine = f_main + gate * f_diff

        pred_body = self.head_body(self.activation(self.head_body_norm(body_feat)))
        pred_bound = self.head_bound(self.activation(self.head_bound_norm(bound_feat)))
        pred_aux = self.aux_head(f_aux)
        pred_final = self.head_final(self.activation(self.head_final_norm(f_refine)))

        return pred_body, pred_bound, pred_aux, pred_final

if __name__ == '__main__':
    torch.set_num_threads(1)
    model = BCRNet(
        img_channels=1,
        base_channels=16,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        dropout=0.2,
        norm='gn',
        num_groups=16,
    )
    x = torch.randn(2, 1, 256, 256)
    outs = model(x)
    print('Number of outputs:', len(outs))
    for i, out in enumerate(outs):
        print(f'Output {i} shape:', out.shape)
