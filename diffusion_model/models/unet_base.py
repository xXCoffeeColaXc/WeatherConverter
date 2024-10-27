import torch
import torch.nn as nn


def get_time_embedding(time_steps, temb_dim):
    r"""
    Convert time steps tensor into an embedding using the
    sinusoidal time embedding formula
    :param time_steps: 1D tensor of length batch size
    :param temb_dim: Dimension of the embedding
    :return: BxD embedding representation of B time steps
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"

    # factor = 10000^(2i/d_model)
    factor = 10000**(torch.arange(0, temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))

    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


class DownBlock(nn.Module):
    r"""
    Down conv block with attention.
    Sequence of following block
    1. Resnet block with time embedding
    2. Attention block
    3. Downsample using 2x2 average pooling
    """

    def __init__(self, in_channels, out_channels, t_emb_dim, down_sample=True, num_heads=4, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(
                        in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1
                    ),
                ) for i in range(num_layers)
            ]
        )
        self.t_emb_layers = nn.ModuleList(
            [nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_channels)) for _ in range(num_layers)]
        )
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                ) for _ in range(num_layers)
            ]
        )
        self.attention_norms = nn.ModuleList([nn.GroupNorm(8, out_channels) for _ in range(num_layers)])

        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)]
        )
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, 4, 2, 1) if self.down_sample else nn.Identity()

    def forward(self, x, t_emb):
        out = x
        for i in range(self.num_layers):

            # Resnet block of Unet
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

            # Attention block of Unet
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn

        out = self.down_sample_conv(out)
        return out


class MidBlock(nn.Module):
    r"""
    Mid conv block with attention.
    Sequence of following blocks
    1. Resnet block with time embedding
    2. Attention block
    3. Resnet block with time embedding
    """

    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads=4, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(
                        in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1
                    ),
                ) for i in range(num_layers + 1)
            ]
        )
        self.t_emb_layers = nn.ModuleList(
            [nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_channels)) for _ in range(num_layers + 1)]
        )
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                ) for _ in range(num_layers + 1)
            ]
        )

        self.attention_norms = nn.ModuleList([nn.GroupNorm(8, out_channels) for _ in range(num_layers)])

        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)]
        )
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers + 1)
            ]
        )

    def forward(self, x, t_emb):
        out = x

        # First resnet block
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)

        for i in range(self.num_layers):

            # Attention Block
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn

            # Resnet Block
            resnet_input = out
            out = self.resnet_conv_first[i + 1](out)
            out = out + self.t_emb_layers[i + 1](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i + 1](out)
            out = out + self.residual_input_conv[i + 1](resnet_input)

        return out


class UpBlock(nn.Module):
    r"""
    Up conv block with attention.
    Sequence of following blocks
    1. Upsample
    1. Concatenate Down block output
    2. Resnet block with time embedding
    3. Attention Block
    """

    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample=True, num_heads=4, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(
                        in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1
                    ),
                ) for i in range(num_layers)
            ]
        )
        self.t_emb_layers = nn.ModuleList(
            [nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_channels)) for _ in range(num_layers)]
        )
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                ) for _ in range(num_layers)
            ]
        )

        self.attention_norms = nn.ModuleList([nn.GroupNorm(8, out_channels) for _ in range(num_layers)])

        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)]
        )
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                                                 4, 2, 1) \
            if self.up_sample else nn.Identity()

    def forward(self, x, out_down, t_emb):
        x = self.up_sample_conv(x)
        x = torch.cat([x, out_down], dim=1)

        out = x
        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn

        return out


class Unet(nn.Module):
    r"""
    Unet model comprising
    Down blocks, Midblocks and Uplocks
    """

    def __init__(self, model_config):
        super().__init__()
        im_channels = model_config['im_channels']
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.t_emb_dim = model_config['time_emb_dim']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        self.num_heads = model_config.get('num_heads', 4)
        self.final_channels = model_config.get('final_channels', 16)  # Parameterized

        # Validations
        assert self.mid_channels[0] == self.down_channels[-1], "First mid_channels must match last down_channels"
        assert self.mid_channels[-1] == self.down_channels[-2], "Last mid_channels must match second last down_channels"
        assert len(self.down_sample) == len(self.down_channels) - 1, "down_sample length must be len(down_channels)-1"

        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim), nn.SiLU(), nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )

        # Input convolution
        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=1)

        # Downsampling blocks
        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.downs.append(
                DownBlock(
                    self.down_channels[i],
                    self.down_channels[i + 1],
                    self.t_emb_dim,
                    down_sample=self.down_sample[i],
                    num_heads=model_config.get('num_heads', 4),
                    num_layers=self.num_down_layers
                )
            )

        # Mid blocks
        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(
                MidBlock(
                    self.mid_channels[i],
                    self.mid_channels[i + 1],
                    self.t_emb_dim,
                    num_heads=model_config.get('num_heads', 4),
                    num_layers=self.num_mid_layers
                )
            )

        # Upsampling blocks
        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels) - 1)):
            in_ch = self.down_channels[i] * 2
            out_ch = self.down_channels[i - 1] if i != 0 else self.final_channels
            self.ups.append(
                UpBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    t_emb_dim=self.t_emb_dim,
                    up_sample=self.down_sample[i],
                    num_heads=model_config.get('num_heads', 4),
                    num_layers=self.num_up_layers
                )
            )

        # Final normalization and convolution
        self.norm_out = nn.GroupNorm(8, self.final_channels)
        self.conv_out = nn.Conv2d(self.final_channels, im_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        # Input projection
        out = self.conv_in(x)  # B x C1 x H x W

        # Time embedding
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)  # B x t_emb_dim

        down_outs = []

        # Downsampling path
        for down in self.downs:
            down_outs.append(out)
            out = down(out, t_emb)
        # down_outs contains outputs before each downsampling

        # Mid blocks
        for mid in self.mids:
            out = mid(out, t_emb)

        # Upsampling path
        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb)

        # Final layers
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)

        return out


if __name__ == '__main__':
    # from diffusion_model.config.models import Config
    # import yaml

    # def load_config(config_path: str) -> Config:
    #     with open(config_path, 'r') as file:
    #         config_data = yaml.safe_load(file)
    #     return Config(**config_data)

    # config = load_config('diffusion_model/config/config.yaml')

    model_config = {
        "im_channels": 3,  # Number of input image channels (e.g., 3 for RGB)
        "im_size": 256,  # Updated image size
        "down_channels": [64, 128, 256, 512],  # Increased channels for deeper architecture
        "mid_channels": [512, 512, 256],  # Updated mid-section channels
        "down_sample": [True, True, False],  # 4 DownBlocks with downsampling
        "time_emb_dim": 128,  # Increased time embedding dimension for larger model
        "num_down_layers": 1,  # Number of ResNet layers per DownBlock
        "num_mid_layers": 1,  # Number of ResNet layers per MidBlock
        "num_up_layers": 1,  # Number of ResNet layers per UpBlock
        "num_heads": 4,  # Increased attention heads for larger feature maps
        "final_channels": 64
    }

    model = Unet(model_config)
    model.to('cuda')
    x = torch.randn(1, 3, 256, 256).to('cuda')
    t = torch.randint(0, 1000, (1,)).to('cuda')

    try:
        output = model(x, t)
        print(output.shape)  # Expected: torch.Size([1, 3, 256, 256])
    except Exception as e:
        print(e)
