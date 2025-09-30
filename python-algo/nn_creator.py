import torch
import torch.nn as nn

class UnitFeatureNet(nn.Module):
    def __init__(self):
        super(UnitFeatureNet, self).__init__()

        # Unit 1D-CNN
        # Start: (8, 28)
        self.unit_features = nn.Sequential(
            nn.Conv1d(8, 32, 5), # (32, 24)
            nn.LeakyReLU(),
            nn.Conv1d(32, 64, 5), # (64, 20)
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(64, 96, 5), # (96, 16)
            nn.BatchNorm1d(96),
            nn.LeakyReLU(),
            nn.Conv1d(96, 128, 5), # (128, 12)
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Conv1d(128, 160, 5), # (160, 8)
            nn.BatchNorm1d(160),
            nn.LeakyReLU(),
            nn.Conv1d(160, 192, 5), # (192, 4)
            nn.BatchNorm1d(192),
            nn.LeakyReLU(),
            nn.Conv1d(192, 224, 4), # (224, 1)
            nn.BatchNorm1d(224),
            nn.LeakyReLU(),
            nn.Flatten(-2, -1)
        )
        # End: (224)

    def forward(self, unit_input):
        return self.unit_features.forward(unit_input)

class UnitAgent(nn.Module):
    def __init__(self):
        super(UnitAgent, self).__init__()

        # Start: (8, 28)
        self.unit_features = UnitFeatureNet()
        # End : (224)
        
        # Start: (12, 29, 15)
        self.all_building_features = nn.Sequential(
            nn.Conv2d(12, 16, (3, 3), 1, 1), # (16, 29, 15)
            nn.LeakyReLU(),
            nn.Conv2d(16, 24, (5, 5), 1, 1), # (24, 27, 13)
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),
            nn.Conv2d(24, 32, (7, 7), 1, 1), # (32, 23, 9)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 40, (7, 7), 1, 1), # (40, 19, 5)
            nn.BatchNorm2d(40),
            nn.LeakyReLU(),
            nn.Flatten(-3, -1),
            nn.Linear(3800, 604), # Dimensionality reduction
            nn.LeakyReLU()
        )
        # End: (604)

        # Start: (30, 28)
        self.unit_action_net = nn.Sequential(
            nn.ConvTranspose1d(30, 30, 5, 1, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(30, 30, 5, 1, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(30, 30, 5, 1, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(30, 30, 5, 1, 2),
            nn.Softmax(dim=-2)
        )
        # End: (30, 28)

        # Start: (840)
        self.value_net = nn.Sequential(
            nn.Linear(840, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1)
        )
        # End: (1)

    def forward(self, unit_input, all_building_input, stats):
        '''
        Input (batch_size, ...) and use BS = 1 for inference.
        '''
        batch_size = stats.shape[0]
        unit_feature_map = self.unit_features.forward(unit_input)
        building_feature_map = self.all_building_features.forward(all_building_input)

        combined = torch.cat((unit_feature_map, building_feature_map, stats), dim=-1)
        map_combined = combined.reshape((batch_size, 30, 28))

        unit_action_probs = self.unit_action_net.forward(map_combined)
        unit_action_probs_transpose = torch.transpose(unit_action_probs, -2, -1) # (BS, 28, 30)
        unit_action_probs_map = unit_action_probs_transpose.reshape((batch_size, 28, 3, 10))

        value = self.value_net.forward(combined)
        return unit_action_probs_map, value
    
    def count_parameters(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
class BuildingAgent(nn.Module):
    def __init__(self):
        super(BuildingAgent, self).__init__()

        # Start: (8, 28)
        self.unit_features = UnitFeatureNet()
        # End : (224)

        # Start: (210, 12)
        self.initial_projetcion_layer = nn.Sequential(
            nn.Linear(12, 32),
            nn.LeakyReLU()
        )
        # End: (210, 32)
        self.initial_pos_embedding = nn.Parameter(torch.rand(210, 32))
        initial_encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dropout=0, batch_first=True)

        # Start: (210, 32)
        self.building_features = nn.Sequential(
            nn.TransformerEncoder(initial_encoder_layer, num_layers=3),
            nn.Flatten(-2, -1),
            nn.Linear(6720, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 2284),
            nn.LeakyReLU()
        )
        # End: (2284)

        # Start: (210, 12)
        self.projection_layer = nn.Sequential(
            nn.Linear(12, 32),
            nn.LeakyReLU()
        )
        # End: (210, 32)
        self.pos_embedding = nn.Parameter(torch.rand(210, 32))
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dropout=0, batch_first=True)

        # Start: (210, 32)
        self.building_action_net = nn.Sequential(
            nn.TransformerEncoder(encoder_layer, num_layers=3),
            nn.Linear(32, 12),
            nn.Softmax(dim=-1)
        )
        # End: (210, 12)

        # Start: (2520)
        self.value_net = nn.Sequential(
            nn.Linear(2520, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
        # End: (1)
        
    def forward(self, unit_input, my_building_input, stats):
        '''
        Input (batch_size, ...) and use BS = 1 for inference.
        '''
        batch_size = stats.shape[0]
        unit_feature_map = self.unit_features.forward(unit_input)
        building_proj = self.initial_projetcion_layer.forward(my_building_input)
        building_embed = building_proj + self.initial_pos_embedding
        building_feature_map = self.building_features.forward(building_embed)

        combined = torch.cat((unit_feature_map, building_feature_map, stats), dim=-1)
        map_combined = combined.reshape((batch_size, 210, 12))
        projected = self.projection_layer.forward(map_combined)
        embedded = projected + self.pos_embedding

        building_action_probs = self.building_action_net.forward(embedded)
        building_action_probs_map = building_action_probs.reshape((batch_size, 210, 6, 2))

        value = self.value_net.forward(combined)
        return building_action_probs_map, value
    
    def count_parameters(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)

class TerminalA2C(nn.Module):
    def __init__(self, embed_dim=32, transformer_depth=3):
        super(TerminalA2C, self).__init__()

        # CNN/Transformer feature extractor for map
        self.encoder = nn.Sequential(
            nn.Conv2d(12, embed_dim // 2, kernel_size=3, padding=1),  # [B, 32, 28, 28]
            nn.LeakyReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, padding=1),       # [B, embed_dim, 28, 28]
            nn.LeakyReLU(),
        )

        # Positional embeddings (learned 2D)
        self.pos_embed = nn.Parameter(torch.randn(28 * 28, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dropout=0, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)

        self.unit_obs = nn.Sequential(
            nn.Linear(56, embed_dim),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=-2),
            nn.Linear(embed_dim * 3, 20),
            nn.LeakyReLU()
        )


        self.buildings = nn.Sequential(
            nn.Linear(embed_dim + 32, embed_dim * 3),
            nn.LeakyReLU(),
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, 9),
            nn.Softmax(dim=-1)
        )

        self.units = nn.Sequential(
            nn.Linear(embed_dim + 32, embed_dim + 32),
            nn.LeakyReLU(),
            nn.Linear(embed_dim + 32, embed_dim + 32),
            nn.LeakyReLU(),
            nn.Linear(embed_dim + 32, 28 * 3 * 15),
            nn.Softmax(dim=-1)
        )

        self.value = nn.Sequential(
            nn.Linear(embed_dim + 32, embed_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, mu, tu, ms, msc, ts, tsc, board):
        
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

        all_units = torch.cat((mu, tu), dim=-1)
        all_units = self.unit_obs.forward(all_units)

        all_stats = torch.cat((all_units, ms, msc, ts, tsc), dim=-1)

        global_fused = torch.cat([pooled_board_features, all_stats], dim=-1)
        with open("fun.txt", "w") as file:
            file.write(str(mu.shape) + "       " + str(all_stats.shape))

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

    model0 = UnitAgent()
    model1 = BuildingAgent()


    print("Unit model:", model0.count_parameters())
    print("Building model:", model1.count_parameters())

    model0.eval()
    model1.eval()

    # Tests
    units = torch.rand((1, 7, 28))
    all_buildings = torch.rand((1, 12, 29, 15))
    my_buildings = torch.rand((1, 210, 12))
    stats = torch.rand((1, 12))

    import time
    start = time.time()
    _ = model0.forward(units, all_buildings, stats)
    _ = model1.forward(units, my_buildings, stats)
    print(f"Passed in : {time.time() - start} seconds.")