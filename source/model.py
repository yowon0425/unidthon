import torch
import torch.nn as nn
import torch.nn.functional as F

class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * 7 - 1) * (2 * 7 - 1), num_heads))
        coords_h = torch.arange(7)
        coords_w = torch.arange(7)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += 7 - 1
        relative_coords[:, :, 1] += 7 - 1
        relative_coords[:, :, 0] *= 2 * 7 - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1).contiguous())
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            7 * 7, 7 * 7, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = torch.softmax(attn * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
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
        return x * y.expand_as(x)

class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1, groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)
        self.se = SELayer(channels)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        x = self.se(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous().reshape(b, c, h, w))
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous().reshape(b, c, h, w))
        return x

class AdaptiveFusionModule(nn.Module):
    def __init__(self, channels):
        super(AdaptiveFusionModule, self).__init__()
        self.fc1 = nn.Conv2d(channels*2, channels, kernel_size=1)
        self.fc2 = nn.Conv2d(channels*2, channels, kernel_size=1)

    def forward(self, x1, x2):
        fusion = torch.cat([x1, x2], dim=1)
        attention = torch.sigmoid(self.fc1(fusion))
        return self.fc2(fusion) * attention + x1 * (1 - attention)

class Restormer(nn.Module):
    def __init__(self, num_blocks=[4,6,6,8], num_heads=[1,2,4,8], channels=[24,48,96,192], num_refinement=4, expansion_factor=2.66):
        super(Restormer, self).__init__()
        
        self.embed_conv = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)
        
        self.encoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(num_ch, num_ah, expansion_factor) for _ in range(num_tb)]) for num_tb, num_ah, num_ch in zip(num_blocks, num_heads, channels)])
        self.downs = nn.ModuleList([nn.Conv2d(num_ch, num_ch*2, kernel_size=4, stride=2, padding=1) for num_ch in channels[:-1]])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(num_ch, num_ch//2, kernel_size=2, stride=2) for num_ch in reversed(channels)[:-1]])
        self.reduces = nn.ModuleList([AdaptiveFusionModule(num_ch) for num_ch in reversed(channels)[1:]])
        
        self.decoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(channels[2], num_heads[2], expansion_factor) for _ in range(num_blocks[2])])])
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[1], expansion_factor) for _ in range(num_blocks[1])]))
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor) for _ in range(num_blocks[0])]))
        
        self.refinement = nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor) for _ in range(num_refinement)])
        self.output = nn.Conv2d(channels[1], 3, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        fo = self.embed_conv(x)
        out_enc1 = self.encoders[0](fo)
        out_enc2 = self.encoders[1](self.downs[0](out_enc1))
        out_enc3 = self.encoders[2](self.downs[1](out_enc2))
        out_enc4 = self.encoders[3](self.downs[2](out_enc3))

        out_dec3 = self.decoders[0](self.reduces[0](self.ups[0](out_enc4), out_enc3))
        out_dec2 = self.decoders[1](self.reduces[1](self.ups[1](out_dec3), out_enc2))
        fd = self.decoders[2](self.reduces[2](self.ups[2](out_dec2), out_enc1))
        
        fr = self.refinement(fd)
        out = self.output(fr) + x
        return out

def progressive_learning(model, train_loader, criterion, optimizer, num_epochs, patch_sizes=[128, 192, 256]):
    for epoch in range(num_epochs):
        patch_size = patch_sizes[min(epoch // (num_epochs // len(patch_sizes)), len(patch_sizes) - 1)]
        for batch in train_loader:
            # Crop input to current patch size
            inputs = F.interpolate(batch['input'], size=(patch_size, patch_size), mode='bilinear', align_corners=False)
            targets = F.interpolate(batch['target'], size=(patch_size, patch_size), mode='bilinear', align_corners=False)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Patch Size: {patch_size}")

# Example usage
model = Restormer()
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
train_loader = ...  # Your data loader here
progressive_learning(model, train_loader, criterion, optimizer, num_epochs=100)