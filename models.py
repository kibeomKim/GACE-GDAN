import torch
import torch.nn as nn
import torch.nn.functional as F


class Feature_Extractor(nn.Module):
    def __init__(self):
        super(Feature_Extractor, self).__init__()
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32, track_running_stats=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(32, track_running_stats=False)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(64, track_running_stats=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(64, track_running_stats=False)

        self.fc = nn.Linear(64*3*3, 256)

    def forward(self, state):

        x = state

        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = F.relu(self.batchnorm3(self.conv3(x)))
        x = F.relu(self.batchnorm4(self.conv4(x)))

        x = x.view(x.size(0), -1)
        img_feat = F.relu(self.fc(x))

        return img_feat


class Actor_Critic(nn.Module):
    def __init__(self):
        super(Actor_Critic, self).__init__()
        self.embedding = nn.Embedding(4, 25)
        self.target_att_linear = nn.Linear(25, 256)

        self.lstm = nn.LSTMCell(512, 256)

        self.layer_query = nn.Linear(256, 256)
        self.layer_key = nn.Linear(256, 256)

        self.before_attn = nn.Linear(512, 512)
        self.attn = nn.Linear(512, 256)

        self.mlp_policy1 = nn.Linear(256, 128)
        self.mlp_policy2 = nn.Linear(128, 64)
        self.policy = nn.Linear(64, 3)

        self.mlp_value1 = nn.Linear(256, 64)
        self.mlp_value2 = nn.Linear(64, 32)  # 64
        self.value = nn.Linear(32, 1)

    def forward(self, img_feat, instruction_idx, _hx, _cx, query):
        word_embedding = self.embedding(instruction_idx)
        word_embedding = word_embedding.view(word_embedding.size(0), -1)

        word_embedding = self.target_att_linear(word_embedding)
        gated_att = torch.sigmoid(word_embedding)  # gated_att

        # apply gated attention
        gated_fusion = torch.mul(img_feat, gated_att)
        lstm_input = torch.cat([gated_fusion, gated_att], 1)

        hx, cx = self.lstm(lstm_input, (_hx, _cx))

        mlp_input = torch.cat([gated_fusion, hx], 1)

        mlp_attn = F.relu(self.before_attn(mlp_input))
        key, value = torch.chunk(mlp_attn, 2, dim=1)

        w_q = F.relu(self.layer_query(query))
        w_k = F.relu(self.layer_key(key))
        u_t = torch.tanh(w_q + w_k)
        attention_vector = torch.mul(u_t, value)

        attn_weight = torch.cat([attention_vector, hx], 1)
        attn_weight = F.relu(self.attn(attn_weight))

        policy = F.relu(self.mlp_policy1(attn_weight))
        policy = F.relu(self.mlp_policy2(policy))

        value = F.relu(self.mlp_value1(attn_weight))
        value = F.relu(self.mlp_value2(value))

        return self.value(value), self.policy(policy), hx, cx


class Goal_discriminator(nn.Module):
    def __init__(self):
        super(Goal_discriminator, self).__init__()

        self.linear1 = nn.Linear(256, 256)
        self.linear2 = nn.Linear(256, 4)

    def forward(self, img_feat):
        query = F.relu(self.linear1(img_feat))
        x = self.linear2(query)

        return x, query


class GDAN(nn.Module):
    def __init__(self):
        super(GDAN, self).__init__()
        self.sigma = Feature_Extractor()

        self.d = Goal_discriminator()

        self.f = Actor_Critic()

    def forward(self, state, instruction, _hx, _cx, use_ActorCritic=True):
        preds, value, policy, hx, cx = None, None, None, None, None

        e_t = self.sigma(state)
        preds, query = self.d(e_t)

        if use_ActorCritic:
            value, policy, hx, cx = self.f(e_t, instruction, _hx, _cx, query)


        return preds, value, policy, hx, cx
