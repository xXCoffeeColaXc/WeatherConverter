import torch
import torch.nn as nn
from diffusion_model_v2.config.models import ModelConfig
from utils import print_gpu_memory


def get_time_embedding(time_steps, temb_dim):
    """
    Convert time steps tensor into an embedding using the
    sinusoidal time embedding formula

    Args:
        time_steps (torch.Tensor): 1D tensor of length batch size
        temb_dim (int): Dimension of the embedding

    Returns:
        torch.Tensor: BxD embedding representation of B time steps
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"

    # factor = 10000^(2i/d_model)
    factor = 10000**(
        (torch.arange(start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )

    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


class EfficientAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super(EfficientAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.scale = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        B, N, C = x.shape
        qkv = self.qkv_proj(x)  # [B, N, 3C]
        qkv = qkv.view(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is [B, num_heads, N, head_dim]

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn_weights = attn_weights.softmax(dim=-1)
        attn_output = attn_weights @ v  # [B, num_heads, N, head_dim]

        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(attn_output)
        return out


class DownBlock(nn.Module):

    def __init__(self, in_channels, out_channels, t_emb_dim, down_sample=True, num_heads=4, num_layers=1):
        """
        Down conv block with attention.
            1. Resnet block with time embedding
            2. Attention block
            3. Downsample using 2x2 average pooling

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            t_emb_dim (int): Dimension of the time embedding.
            down_sample (bool, optional): Whether to perform downsampling using 2x2 average pooling. Defaults to True.
            num_heads (int, optional): Number of attention heads. Defaults to 4.
            num_layers (int, optional): Number of layers in the block. Defaults to 1.
        """
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

        self.attentions = nn.ModuleList([EfficientAttention(out_channels, num_heads) for _ in range(num_layers)])
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, 4, 2, 1) if self.down_sample else nn.Identity()

    def forward(self, x, t_emb):
        """
        Forward pass of the DownBlock module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            t_emb (torch.Tensor): Time embedding tensor of shape (batch_size, t_emb_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height', width').
        """
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
            out_attn = self.attentions[i](in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn

        out = self.down_sample_conv(out)
        return out


class MidBlock(nn.Module):

    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads=4, num_layers=1):
        """
        Mid conv block with attention.
            Sequence of following blocks
            1. Resnet block with time embedding
            2. Attention block
            3. Resnet block with time embedding

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            t_emb_dim (int): Dimension of the time embedding.
            num_heads (int, optional): Number of attention heads. Defaults to 4.
            num_layers (int, optional): Number of layers. Defaults to 1.
        """
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

        self.attentions = nn.ModuleList([EfficientAttention(out_channels, num_heads) for _ in range(num_layers)])
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers + 1)
            ]
        )

    def forward(self, x, t_emb):
        """
        Forward pass of the MidBlock module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            t_emb (torch.Tensor): Time embedding tensor of shape (batch_size, t_emb_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
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
            out_attn = self.attentions[i](in_attn)
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

    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample=True, num_heads=4, num_layers=1):
        """
        Up conv block with attention.
            1. Upsample
            2. Concatenate Down block output
            3. Resnet block with time embedding
            4. Attention Block

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            t_emb_dim (int): Dimension of the time embedding.
            up_sample (bool, optional): Whether to perform upsampling. Defaults to True.
            num_heads (int, optional): Number of attention heads. Defaults to 4.
            num_layers (int, optional): Number of layers in the block. Defaults to 1.
        """
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

        self.attentions = nn.ModuleList([EfficientAttention(out_channels, num_heads) for _ in range(num_layers)])
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels //
                                                 2, 4, 2, 1) if self.up_sample else nn.Identity()

    def forward(self, x, out_down, t_emb):
        """
        Forward pass of the UpBlock module.

        Args:
            x (torch.Tensor): Input tensor.
            out_down (torch.Tensor): Output tensor from the down block.
            t_emb (torch.Tensor): Time embedding tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
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
            out_attn = self.attentions[i](in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn

        return out


class Unet(nn.Module):
    r"""
    Unet model comprising
    Down blocks, Midblocks and Uplocks
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        im_channels = model_config.im_channels
        self.down_channels = model_config.down_channels
        self.mid_channels = model_config.mid_channels
        self.t_emb_dim = model_config.time_emb_dim
        self.down_sample = model_config.down_sample
        self.num_down_layers = model_config.num_down_layers
        self.num_mid_layers = model_config.num_mid_layers
        self.num_up_layers = model_config.num_up_layers

        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1

        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim), nn.SiLU(), nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )

        self.up_sample = list(reversed(self.down_sample))
        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=(1, 1))

        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.downs.append(
                DownBlock(
                    self.down_channels[i],
                    self.down_channels[i + 1],
                    self.t_emb_dim,
                    down_sample=self.down_sample[i],
                    num_layers=self.num_down_layers
                )
            )

        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(
                MidBlock(
                    self.mid_channels[i], self.mid_channels[i + 1], self.t_emb_dim, num_layers=self.num_mid_layers
                )
            )

        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels) - 1)):
            self.ups.append(
                UpBlock(
                    self.down_channels[i] * 2,
                    self.down_channels[i - 1] if i != 0 else self.down_channels[0],
                    self.t_emb_dim,
                    up_sample=self.down_sample[i],
                    num_layers=self.num_up_layers
                )
            )

        self.norm_out = nn.GroupNorm(8, self.down_channels[0])
        self.conv_out = nn.Conv2d(self.down_channels[0], im_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        # Shapes assuming downblocks are [C1, C2, C3, C4]
        # Shapes assuming midblocks are [C4, C4, C3]
        # Shapes assuming downsamples are [True, True, False]
        # B x C x H x W
        out = self.conv_in(x)
        print("Shape after conv_in:", out.shape)
        # B x C1 x H x W

        # t_emb -> B x t_emb_dim
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)

        down_outs = []

        for idx, down in enumerate(self.downs):
            down_outs.append(out)
            out = down(out, t_emb)
            print(f"Shape after downblock {idx+1}:", out.shape)
        # down_outs  [B x C1 x H x W, B x C2 x H/2 x W/2, B x C3 x H/4 x W/4]
        # out B x C4 x H/4 x W/4

        for mid in self.mids:
            out = mid(out, t_emb)
            print("Shape after midblock:", out.shape)
        # out B x C3 x H/4 x W/4

        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb)
            print("Shape after upblock:", out.shape)
            # out [B x C2 x H/4 x W/4, B x C1 x H/2 x W/2, B x 16 x H x W]
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        # out B x C x H x W
        print("Final output shape:", out.shape)
        return out


if __name__ == '__main__':
    from diffusion_model_v2.config.models import Config
    import yaml

    def load_config(config_path: str) -> Config:
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        return Config(**config_data)

    config = load_config('diffusion_model_v2/config/config.yaml')

    model = Unet(config.model)
    print_gpu_memory("After model initialization")

    input_tensor = torch.randn(1, config.model.im_channels, config.model.im_size, config.model.im_size)

    t = (10,)

    try:
        print_gpu_memory("Before forward pass")
        output = model(input_tensor, t)
        print(output.shape)
        print_gpu_memory("After forward pass")
    except Exception as e:
        print(e)
