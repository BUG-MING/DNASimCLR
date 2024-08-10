import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50

class DNN_model(nn.Module):
    def __init__(self, feature_dim=5):
        super(DNN_model, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False),
                               nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True),
                               nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(out, dim=-1)
#
# class DNN_model(nn.Module):
#     def __init__(self,num_fature,dropout=0.0):
#         super(DNN_model, self).__init__()
#         self.num_fature=num_fature
#         self.model=nn.Sequential(
#             nn.Linear(num_fature,num_fature*4),
#             nn.BatchNorm1d(num_fature*4),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             # nn.Linear(num_fature * 4, num_fature * 16),
#             # nn.BatchNorm1d(num_fature * 16),
#             # nn.ReLU(),
#             # nn.Dropout(dropout),
#             # nn.Linear(num_fature * 16, num_fature*4),
#             # nn.BatchNorm1d(num_fature * 4),
#             # nn.ReLU(),
#             # nn.Dropout(dropout),
#             nn.Linear(num_fature * 4, num_fature),
#             nn.BatchNorm1d(num_fature),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(num_fature, 5)
#         )
#
#     def forward(self, x):
#         result = self.model(x)
#         return result


