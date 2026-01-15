from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

class VICRegLoss(nn.Module):
    def __init__(self, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, x, y):
        # 1. Invariance Loss (Mean Squared Error)
        sim_loss = F.mse_loss(x, y)

        # 2. Variance Loss
        std_x = torch.sqrt(x.var(dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(1.0 - std_x))

        # 3. Covariance Loss
        batch_size, num_features = x.shape
        x = x - x.mean(dim=0)
        cov_x = (x.T @ x) / (batch_size - 1)
        diag = torch.eye(num_features, device=x.device)
        cov_loss = (cov_x * (1 - diag)).pow(2).sum() / num_features

        return (self.sim_coeff * sim_loss + 
                self.std_coeff * std_loss + 
                self.cov_coeff * cov_loss)

class DINOLoss(nn.Module):
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center = nn.Parameter(torch.zeros(1, out_dim))
        
    def forward(self, student_output, teacher_output):
        student_out = student_output / self.student_temp
        
        teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)
        teacher_out = teacher_out.detach()
        
        student_out = F.log_softmax(student_out, dim=-1)
        
        loss = -torch.sum(teacher_out * student_out, dim=-1).mean()
        
        self.update_center(teacher_output)
        
        return loss
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center.data = self.center.data * 0.9 + batch_center * 0.1
