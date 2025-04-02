import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from rdkit.Chem import AllChem


class Agent(nn.Module):
    def __init__(self, feat_dim, hidden_size):
        super(Agent, self).__init__()
        self.affine1 = nn.Linear(feat_dim + 2 + 6, hidden_size)
        self.dropout = nn.Dropout(p=0.5)
        self.affine2 = nn.Linear(hidden_size, 2)
        self.saved_log_probs = {}

    def forward(self, x, geometric_features):
        x = torch.cat([x, geometric_features], dim=1)
        x = self.affine1(x)
        x = F.relu(x)
        scores = self.affine2(x)
        return F.softmax(scores, dim=1)


def sample(agent, subgraph_feature, geometric_feature, iter_num, sample_number):
    prob = agent(subgraph_feature, geometric_feature)
    m = Categorical(prob)
    a = m.sample()
    take_action = (np.sum(a.cpu().numpy()) != 0)
    if take_action:
        if sample_number not in agent.saved_log_probs:
            agent.saved_log_probs[sample_number] = {}
        if iter_num not in agent.saved_log_probs[sample_number]:
            agent.saved_log_probs[sample_number][iter_num] = [m.log_prob(a)]
        else:
            agent.saved_log_probs[sample_number][iter_num].append(m.log_prob(a))
    return a.cpu().numpy(), take_action
    
