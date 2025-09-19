import torch
import torch.nn as nn

class TerminalA2C(nn.Module):
    def __init__(self, embed_dim=32, transformer_depth=3):
        super(TerminalA2C, self).__init__()

        # CNN/Transformer feature extractor for map
        self.encoder = nn.Sequential(
            nn.Conv2d(12, embed_dim // 2, kernel_size=3, padding=1),  # [B, 32, 28, 28]
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, padding=1),       # [B, embed_dim, 28, 28]
            nn.ReLU(),
        )

        # Positional embeddings (learned 2D)
        self.pos_embed = nn.Parameter(torch.randn(28 * 28, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dropout=0, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)


        self.buildings = nn.Sequential(
            nn.Linear(embed_dim + 12, embed_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, 9),
            nn.Softmax(dim=-1)
        )

        self.units = nn.Sequential(
            nn.Linear(embed_dim + 12, embed_dim + 12),
            nn.LeakyReLU(),
            nn.Linear(embed_dim + 12, embed_dim + 12),
            nn.LeakyReLU(),
            nn.Linear(embed_dim + 12, 28 * 3 * 15),
            nn.Softmax(dim=-1)
        )

        self.value = nn.Sequential(
            nn.Linear(embed_dim + 12, embed_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, ms, msc, ts, tsc, board):
        
        if len(board.shape) > 3:
            batch_size = board.size(0)
        else:
            batch_size = 1

        board_features = self.encoder(board)
        board_features = board_features.flatten(-2)
        board_features = board_features.transpose(-1, -2)
        board_features = board_features + self.pos_embed
        board_features = self.transformer(board_features)


        pooled_board_features = board_features.mean(dim=-2) # Mean pooling

        all_stats = torch.cat((ms, msc, ts, tsc), dim=-1)

        global_fused = torch.cat([pooled_board_features, all_stats], dim=-1)

        local_fused = []
        if len(board_features.shape) == 3:
            for i in range(len(board_features[0])):
                local_fused.append(torch.cat((board_features[:, i, :], all_stats), dim=-1))
        elif len(board_features.shape) == 2:
            for i in range(len(board_features)):
                local_fused.append(torch.cat((board_features[i, :], all_stats), dim=-1))

            
        local_fused = torch.stack(local_fused)


        value = self.value(global_fused)
        building_actions_dist = self.buildings(local_fused)
        building_actions_dist = building_actions_dist.view(batch_size, 784, 3, 3)
        building_actions_dist = building_actions_dist.transpose(1, 2)
        building_actions_dist = building_actions_dist[:, :, :392, :] # [B, 3, 28x14, 3]
        unit_actions_dist = self.units(global_fused)
        unit_actions_dist = unit_actions_dist.view(batch_size, 28, 3, 15)

        return building_actions_dist, unit_actions_dist, value

    def count_parameters(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = TerminalA2C()

    print(model.count_parameters())
    ms, msc, ts, tsc = torch.rand(3), torch.rand(3), torch.rand(3), torch.rand(3)
    board = torch.rand((12, 28, 28))

    a, b, c = model.forward(ms, msc, ts, tsc, board)
    print(a.shape, b.shape, c)