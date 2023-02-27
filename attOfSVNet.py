# Import modules
import torch
import torchvision


class Network(torch.nn.Module):
    def __init__(self, num_channels):
        super(Network, self).__init__()
        self.model1 = torchvision.models.vgg16(pretrained=True)
        self.freeze_module(self.model1)
        self.model1.features[0] = torch.nn.Conv2d(num_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.model2 = torchvision.models.vgg16(pretrained=True)
        self.freeze_module(self.model2)
        self.model2.features[0] = torch.nn.Conv2d(num_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = torch.nn.Linear(2000, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.out = torch.nn.Linear(64, 1)
        
    def freeze_module(self, *modules):
        for m in modules:
            for param in m.parameters():
                param.requires_grad = False
                
    def forward(self, input1, input2):
        model1 = self.model1(input1)
        model2 = self.model2(input2)
        model = torch.cat([model1, model2], axis=1)
        fc1 = self.fc1(model)
        fc2 = self.fc2(fc1)
        fc3 = self.fc3(fc2)
        out = self.out(fc3)
        out = torch.sigmoid(out)
        return out.reshape(-1)
