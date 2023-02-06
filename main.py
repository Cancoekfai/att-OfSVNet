# Import modules
import os
import time
import torch
import random
import sklearn
import argparse
import torchvision
import numpy as np
from tqdm import tqdm
from load_data import Dataset
from attOfSVNet import Network
import matplotlib.pyplot as plt

# Settings
parser = argparse.ArgumentParser(description='Offline Signature Verification')
parser.add_argument('--dataset', type=str, help='dataset name, options: [CEDAR, BHSig-B, BHSig-H]')
parser.add_argument('--epochs', type=int, default=100, help='training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--num_channels', type=int, default=1, help='number of image channels')
parser.add_argument('--image_size', type=tuple, default=(128, 256), help='image size')
parser.add_argument('--optimizer_type', type=str, default='adam', help='optimizer type')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--lrf', type=float, default=0.01, help='learning rate of stopping updates')
parser.add_argument('--cos_lr', type=bool, default=True, help='whether to use the cosine annealing decay method')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
args = parser.parse_args()
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
plt.rcParams['font.sans-serif'] = ['Times New Roman']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_data_path = 'datasets/%s/train.csv'%args.dataset
test_data_path = 'datasets/%s/test.csv'%args.dataset
model_save_path = 'models_%s'%args.dataset


# load data
train_dataset = Dataset(train_data_path, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                   torchvision.transforms.Resize(args.image_size)]))
indices = range(len(train_dataset))
indices = sklearn.utils.shuffle(indices, random_state=seed)
test_dataset = Dataset(test_data_path, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                 torchvision.transforms.Resize(args.image_size)]))
train_db = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                       shuffle=False, sampler=indices)
test_db = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                      shuffle=False)
print('train data: images:', (len(train_dataset), args.num_channels, args.image_size[0], args.image_size[1]), 'labels:', len(train_dataset))
print('test  data: images:', (len(test_dataset), args.num_channels, args.image_size[0], args.image_size[1]), 'labels:', len(test_dataset))


# Modeling
model = Network(args.num_channels).to(device)


# Configure optimizer
optimizer = {
    'adam' : torch.optim.Adam(model.parameters(), args.lr, betas=(args.momentum, 0.999), weight_decay=args.weight_decay),
    'adamw': torch.optim.AdamW(model.parameters(), args.lr, betas=(args.momentum, 0.999), weight_decay=args.weight_decay),
    'sgd'  : torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
}[args.optimizer_type]
if args.cos_lr:
    lf = lambda x: ((1 - np.cos(x * np.pi / args.epochs)) / 2) * (args.lrf - 1) + 1
else:
    lf = lambda x: (1 - x / args.epochs) * (1 - args.lrf) + args.lrf
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

# Model training
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
LOSS = []
LOSS_VAL = []
ACCURACY = []
ACCURACY_VAL = []
val_accuracy_max = 0
t1 = time.time()
for epoch in range(args.epochs):
    Loss = []
    total = 0
    correct = 0
    with tqdm(total=len(train_db), desc='Epoch {}/{}'.format(epoch + 1, args.epochs)) as pbar:
        for step, (x1, x2, y) in enumerate(train_db):
            model.train()
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            output = model(x1, x2)
            # Calculation loss
            BCELoss = torch.nn.BCELoss()
            loss = BCELoss(output, y)
            Loss.append(float(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Calculation accuracy
            pred = torch.where(output >= 0.5, 1, 0)
            total += y.shape[0]
            correct += int((pred == y).sum())
            pbar.set_postfix({'loss': '%.4f'%np.mean(Loss),
                              'accuracy': '%.2f'%((correct/total) * 100) + '%'})
            
            # Model validation
            model.eval()
            if step == len(train_db) - 1:
                Loss_val = []
                total_val = 0
                correct_val = 0
                for x1, x2, y in test_db:
                    x1 = x1.to(device)
                    x2 = x2.to(device)
                    y = y.to(device)
                    output = model(x1, x2)
                    # Calculation loss
                    BCELoss = torch.nn.BCELoss()
                    loss_val = BCELoss(output, y)
                    Loss_val.append(float(loss_val))
                    # Calculation accuracy
                    pred = torch.where(output >= 0.5, 1, 0)
                    total_val += y.shape[0]
                    correct_val += int((pred == y).sum())
                pbar.set_postfix({'loss': '%.4f'%np.mean(Loss),
                                  'val_loss': '%.4f'%np.mean(Loss_val),
                                  'accuracy': '%.2f'%((correct/total) * 100) + '%',
                                  'val_accuracy': '%.2f'%((correct_val/total_val) * 100) + '%'})
            pbar.update(1)
        val_loss = np.mean(Loss_val)
        val_accuracy = (correct_val/total_val) * 100
        torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch%s_%.4f_%.4f.pth'%(epoch, val_loss, val_accuracy)))
        if val_accuracy > val_accuracy_max:
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best.pth'))
            val_accuracy_max = val_accuracy
        scheduler.step()
        LOSS.append(np.mean(Loss))
        LOSS_VAL.append(np.mean(Loss_val))
        ACCURACY.append(correct / total)
        ACCURACY_VAL.append(correct_val / total_val)
train_history = {'loss': LOSS, 'val_loss': LOSS_VAL,
                 'accuracy': ACCURACY, 'val_accuracy': ACCURACY_VAL}
t2 = time.time()
times = t2 - t1
print('Time taken: %d seconds'%times)

def plot_train_history(type_str, train_type, val_type):
    plt.figure(dpi=400)
    plt.plot(train_history[train_type])
    plt.plot(train_history[val_type])
    plt.ylabel(type_str)
    plt.xlabel('Epochs')
    plt.legend(['Training set', 'Test set'], loc='best')
    plt.show()
plot_train_history('Accuracy (%)', 'accuracy', 'val_accuracy')
plot_train_history('Loss value', 'loss', 'val_loss')
