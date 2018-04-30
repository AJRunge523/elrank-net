import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import logging
logger = logging.getLogger(__name__)

class Ranker(nn.Module):
    """Simple neural ranker"""
    def __init__(self, input_size, hidden_size):
        super(Ranker, self).__init__()
        self.input_size  = input_size  #source vocab size
        self.hidden_size = hidden_size

        # The layers of the NN
        self.embed_linear = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(p=0.2)
        # self.intermed_layer = nn.Linear(hidden_size, hidden_size)
        # self.dropout2 = nn.Dropout(p=0.5)
        self.out_layer = nn.Linear(hidden_size, 1)

    def forward(self, input):
        embed = self.dropout1(F.relu(self.embed_linear(input)))
        # intermed = self.dropout2(F.relu(self.intermed_layer(embed)))
        output = self.out_layer(embed)
        return output.squeeze()

    def save(self, fname):
        """Save the model using pytorch's format"""
        logger.info("Saving at: {}".format(fname))
        torch.save(self, fname)

