import torch
import torch.nn as nn
import torch.distributed as dist
from .gather import GatherLayer


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, device, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.world_size = world_size

        self.mask = self.get_mask(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss()
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def get_mask(self, batch_size, world_size):
        mask = torch.ones((batch_size, batch_size), dtype=bool)
        mask = mask.fill_diagonal_(0)
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        sim_ii = self.similarity_f(z_i.unsqueeze(1), z_i.unsqueeze(0)) / self.temperature
        sim_ii = sim_ii[self.mask].reshape(self.batch_size-1, -1)
        sim_jj = self.similarity_f(z_j.unsqueeze(1), z_j.unsqueeze(0)) / self.temperature
        sim_jj = sim_jj[self.mask].reshape(self.batch_size-1, -1)
        sim_ij = self.similarity_f(z_i.unsqueeze(1), z_j.unsqueeze(0)) / self.temperature
        sim_ji = self.similarity_f(z_j.unsqueeze(1), z_i.unsqueeze(0)) / self.temperature
        labels = torch.cat(
            (torch.arange(1, self.batch_size), torch.zeros(self.batch_size))
        ).to(sim_ii.device).long()  # torch.Size([256])
        loss_a = self.criterion(torch.cat((sim_ij, sim_ii), dim=0), labels)
        loss_b = self.criterion(torch.cat((sim_ji, sim_jj), dim=0), labels)
        loss = (loss_a + loss_b)/2
        return loss
