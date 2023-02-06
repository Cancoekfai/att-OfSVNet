# Import modules
import torch


class Encoder(torch.nn.Module):
    def __init__(self, num_channels):
        super(Encoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_channels, 32, 3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.pool3 = torch.nn.MaxPool2d(2, 2)
        self.conv41 = torch.nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.bn41 = torch.nn.BatchNorm2d(64)
        self.conv42 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn42 = torch.nn.BatchNorm2d(128)
        self.pool4 = torch.nn.MaxPool2d(2, 2)
        
    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv1 = self.bn1(conv1)
        conv1 = torch.nn.functional.relu(conv1)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        conv2 = self.bn2(conv2)
        conv2 = torch.nn.functional.relu(conv2)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        conv3 = self.bn3(conv3)
        conv3 = torch.nn.functional.relu(conv3)
        pool3 = self.pool3(conv3)
        conv41 = self.conv41(pool3)
        conv41 = self.bn41(conv41)
        conv41 = torch.nn.functional.relu(conv41)
        conv42 = self.conv42(conv41)
        conv42 = self.bn42(conv42)
        conv42 = torch.nn.functional.relu(conv42)
        concat = torch.cat([conv41, conv42], axis=1)
        pool4 = self.pool4(concat)
        return conv1, conv2, conv3, pool4
    
class Attention2D(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attention2D, self).__init__()
        self.fc1_att = torch.nn.Linear(input_size, hidden_size)
        self.conv_att = torch.nn.Conv2d(input_size, hidden_size, 3, padding=1)
        self.fc2_att = torch.nn.Linear(hidden_size, 1)
        
    def forward(self, img_fvec, patch_fmap):
        g_em = self.fc1_att(img_fvec)
        g_em = g_em.unsqueeze(-1).permute(0, 2, 1)
        x_em = self.conv_att(patch_fmap)
        x_em = x_em.view(x_em.shape[0], -1, x_em.shape[2] * x_em.shape[3]).permute(0, 2, 1)
        actv_sum_feat = torch.tanh(x_em + g_em)
        attn_wts = torch.nn.functional.softmax(self.fc2_att(actv_sum_feat), dim=1).permute(0, 2, 1)
        patch_fmap_ = patch_fmap.view(patch_fmap.shape[0], -1, patch_fmap.shape[2] * patch_fmap.shape[3])
        patch_fmap_ = patch_fmap_.permute(0, 2, 1)
        attn = torch.bmm(attn_wts, patch_fmap_)
        attn = attn.squeeze(1)
        return attn
    
class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = torch.nn.Linear(24576, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.att1 = Attention2D(128, 64)
        self.att2 = Attention2D(64, 32)
        self.att3 = Attention2D(32, 16)
        
    def forward(self, input1, input2, input3, input4):
        flat = torch.flatten(input4, 1)
        fc1 = self.fc1(flat)
        fc1 = torch.nn.functional.relu(fc1)
        fc1 = self.att1(fc1, input3)
        fc2 = self.fc2(fc1)
        fc2 = torch.nn.functional.relu(fc2)
        fc2 = self.att2(fc2, input2)
        fc3 = self.fc3(fc2)
        fc3 = torch.nn.functional.relu(fc3)
        fc3 = self.att3(fc3, input1)
        return fc3
    
class Network(torch.nn.Module):
    def __init__(self, num_channels):
        super(Network, self).__init__()
        self.encoder = Encoder(num_channels)
        self.decoder = Decoder()
        self.out = torch.nn.Linear(64, 1)
        
    def forward(self, input1, input2):
        encoder11, encoder12, encoder13, encoder14 = self.encoder(input1)
        encoder21, encoder22, encoder23, encoder24 = self.encoder(input2)
        decoder1 = self.decoder(encoder11, encoder12, encoder13, encoder14)
        decoder2 = self.decoder(encoder21, encoder22, encoder23, encoder24)
        decoder = torch.cat([decoder1, decoder2], axis=1)
        out = self.out(decoder)
        out = torch.sigmoid(out)
        return out.reshape(-1)
