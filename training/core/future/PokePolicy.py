import torch.nn as nn

# from stable_baselines3.common.policies import register_policy
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3 import PPO
import torch


class PokePolicy(nn.Module):
    def __init__(self, observation_space, action_space, features_dim=512):
        super(PokePolicy, self).__init__()

        # Define the CNN for image input
        self.features_extractor = NatureCNN(observation_space, features_dim)

        # Define the embedding layer for Pokemon types
        self.embedding = nn.Embedding(152, 10)  # 152 types, 10-dimensional embedding

        # Fully connected layer to combine features
        self.fc_combined = nn.Linear(
            features_dim + 60, features_dim
        )  # 10 dim each for 6 Pokemon

        # Policy and value heads
        self.action_net = nn.Sequential(
            nn.Linear(features_dim, 64), nn.ReLU(), nn.Linear(64, action_space.n)
        )
        self.value_net = nn.Sequential(
            nn.Linear(features_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, obs):
        # Separate image and pokemon
        frame = obs["frame"]
        stats = obs["stats"]

        # Extract image features using CNN
        img_features = self.features_extractor(frame)

        # Extract type features using embedding
        # One for each pokemon
        embeddings = []
        for i in range(6):
            embeddings.append(self.embedding(stats[:, i]))

        embeddings_vec = torch.cat(embeddings, dim=1)

        # Combine features
        combined_features = torch.cat([img_features, embeddings_vec], dim=1)
        combined_features = self.fc_combined(combined_features)

        # Policy and value heads
        actions = self.action_net(combined_features)
        values = self.value_net(combined_features)

        return actions, values


# register_policy('PokePolicy', PokePolicy)
