%%capture
!pip install torchsummary
!pip install imblearn

import os
import random
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

import cv2
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from PIL import *
from tqdm import tqdm
from torchsummary import summary
from torchvision import datasets, transforms
from torch.utils.data import  DataLoader, Dataset, ConcatDataset
from sklearn.model_selection import train_test_split

# cuda teknolojisini kullanarak grafik kartında eğitim gerçekleştirmek için

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Data setimizin dosya dizinlerini değişkene atıyoruz

egitim_dosya_dizini= '../input/covid19-veri-seti/Covid19_Veri_Seti/train'
dogrulama_dizini = '../input/covid19-veri-seti/Covid19_Veri_Seti/test'
test_dosya_dizini = '../input/covid19-veri-seti/Covid19_Veri_Seti/val'

# Gelen dataları ön işleme alıyoruz

data_donusum = {
    'train': {
        'dataset1': transforms.Compose([transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomGrayscale(),
            transforms.RandomAffine(translate=(0.05,0.05), degrees=0),
            transforms.ToTensor()
           ]),

        'dataset2' : transforms.Compose([transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomGrayscale(),
            transforms.RandomAffine(translate=(0.1,0.05), degrees=10),
            transforms.ToTensor()

           ]),
        'dataset3' : transforms.Compose([transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomGrayscale(p=1),
            transforms.RandomAffine(translate=(0.08,0.1), degrees=15),
            transforms.ToTensor()
           ]),
    },
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),
}

## Data setinin oluşturulması

dataset1 = datasets.ImageFolder(egitim_dosya_dizini, 
                      transform=data_donusum['train']['dataset1'])

dataset2 = datasets.ImageFolder(egitim_dosya_dizini, 
                      transform=data_donusum['train']['dataset2'])

dataset3 = datasets.ImageFolder(egitim_dosya_dizini, 
                      transform=data_donusum['train']['dataset3'])

norm1, _ = train_test_split(dataset2, test_size= 3875/(1341+3875), shuffle=False)
norm2, _ = train_test_split(dataset3, test_size= 4023/(1341+3875), shuffle=False)

dataset = ConcatDataset([dataset1, norm1, norm2])

egitim_dataseti, val_ds = train_test_split(dataset, test_size=0.3, random_state=2000)

Datasets = {
    'train': egitim_dataseti,
    'test' : datasets.ImageFolder(test_dosya_dizini, data_donusum['test']),
    'val'  : val_ds
}

Dataloaders = {
    'train': DataLoader(Datasets['train'], batch_size = 32, num_workers = 2),
    'test': DataLoader(Datasets['test'], batch_size = 32, shuffle = True, num_workers = 2),
    'val': DataLoader(Datasets['val'], batch_size = 32, shuffle = True, num_workers = 2),
}


dosyalar = []
kategoriler = []
dosya_isimleri = os.listdir(os.path.join(egitim_dosya_dizini,'normal'))
for name in dosya_isimleri:
    dosyalar.append(os.path.join(egitim_dosya_dizini, 'normal', name))
    kategoriler.append('normal')

dosya_isimleri = os.listdir(os.path.join(egitim_dosya_dizini,'covid'))
for name in dosya_isimleri:
    dosyalar.append(os.path.join(egitim_dosya_dizini, 'covid', name))
    kategoriler.append('covid')

random_dosya_index = random.sample(range(len(dosyalar)), 9)
random_fig = plt.figure(figsize = (12, 12))
rows, cols = 3, 3
for i in range(9):
    random_fig.add_subplot(rows, cols, i+1)
    plt.imshow(cv2.imread(dosyalar[random_dosya_index[i]]))
    plt.title(kategoriler[random_dosya_index[i]])
    plt.axis('off')
plt.show()

Tr_COVID = len([label for _, label in Datasets['train'] if label == 1])
Tr_NORMAL = len(Datasets['train']) - Tr_COVID
V_COVID = len([label for _, label in Datasets['val'] if label == 1])
V_NORMAL = len(Datasets['val']) - V_COVID
Te_COVID = len([label for _, label in Datasets['test'] if label == 1])
Te_NORMAL = len(Datasets['test']) - Te_COVID
Cvd = [Tr_COVID, V_COVID, Te_COVID]
No = [Tr_NORMAL, V_NORMAL, Te_NORMAL]
Cvd, No

fig = plt.subplots(figsize =(4, 4))

br1 = np.arange(len(Cvd))
br2 = [x + 0.25 for x in br1]

plt.bar(br1, Cvd, color='r', width = 0.25, label = 'Covid-19')
plt.bar(br2, No, color='b', width = 0.25, label = 'Normal')

plt.ylabel('Toplam Miktar')
plt.xticks([r + 0.25 for r in range(len(Cvd))],
        ['Eğitim', 'Doğrulama', 'Test'])
plt.legend()
plt.show()

# Modellin Tanımlanması

model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(512, 2)
)

model = model.to(device)
summary(model, input_size = (3, 224, 224))


# Eğitim

epochs = 15
alpha = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=alpha)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.1)
loss_fn = nn.CrossEntropyLoss()

def trainer_loop(model, trainloader, loss_fn, optimizer, scheduler = None, t_gpu = False):
    model.train()
    tr_loss, tr_acc = 0.0, 0.0
    for i, data in enumerate(tqdm(trainloader)):
        img, label = data
        if t_gpu:
                img, label = img.cuda(), label.cuda()
        optimizer.zero_grad()
        output = model(img)
        _, pred = torch.max(output.data, 1)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        
        tr_loss += loss.item()
        tr_acc += torch.sum(pred == label.data)
        torch.cuda.empty_cache()

    scheduler.step() if scheduler != None else None
    return tr_loss/len(trainloader.dataset), 100*tr_acc/len(trainloader.dataset)
    
def val_loop(model, val_loader, loss_fn, t_gpu=False):
    model.train(False)
    model.eval()
    val_loss, val_acc = 0.0, 0.0
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader)):
            img, label = data
            if t_gpu:
                    img, label = img.cuda(), label.cuda()
            output = model(img)
            _, pred = torch.max(output.data, 1)
            loss = loss_fn(output, label)

            val_loss += loss.item()
            val_acc += torch.sum(pred == label.data)

    return val_loss/len(val_loader.dataset), 100*val_acc/len(val_loader.dataset)

def train_model(epochs, model, trainloader, valloader, loss_fn, optimizer, scheduler = None, t_gpu = False):
    stat_dict = {
        'learning_rate':[],
        'train_loss':    [],
        'train_acc':     [],
        'val_loss':      [],
        'val_acc':       []    
    }
    print('*'*5+'Eğitim Başladı'+'*'*5)
    for ep in range(epochs):
        print(f'Eğitim turu: {ep+1}')
        t_loss, t_acc = trainer_loop(
            model, trainloader, loss_fn, optimizer, scheduler, t_gpu
        )
        v_loss, v_acc = val_loop(
            model, valloader, loss_fn, t_gpu
        )
        print(f'Öğrenme Oranı: {optimizer.param_groups[0]["lr"]}')
        print(f'Eğitim   : Kayıp: {t_loss}    Kesinlik: {t_acc}%')
        print(f'Doğrulama : Kayıp: {v_loss}    Kesinlik: {v_acc}%')
        stat_dict['learning_rate'].append(optimizer.param_groups[0]["lr"])
        stat_dict['train_loss'].append(t_loss)
        stat_dict['val_loss'].append(v_loss)
        stat_dict['train_acc'].append(t_acc)
        stat_dict['val_acc'].append(v_acc)
    print('Eğitim Tamamlandı')
    return stat_dict

hist = train_model(epochs, model, Dataloaders['train'], Dataloaders['val'], loss_fn, optimizer,  scheduler, device == 'cuda')

### Doğruluk

fig, ax = plt.subplots(figsize=(8,5))

AT = ax.plot(np.linspace(1, epochs, epochs), hist['train_acc'], 'g-', label='Train Acc')
AV = ax.plot(np.linspace(1, epochs, epochs), hist['val_acc'], 'y-', label='Val Acc')

ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')

ax.set_xlim([1, (len(hist['val_acc']))])
if len(hist['val_acc']) >= 30:
    ax.set_xticks(range(1, (len(hist['val_acc'])+1), 5))
elif len(hist['val_acc']) >= 20:
    ax.set_xticks(range(1, (len(hist['val_acc'])+1), 2))
elif len(hist['val_acc']) < 20:
    ax.set_xticks(range(1, (len(hist['val_acc'])+1)))

lns = AT+AV
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='upper right')
ax.grid('on')

# Modelin test edilmesi

@torch.no_grad()
def test_loop(model, testdata, loss_fn, t_gpu):
    print('*'*5+'Test Başladı'+'*'*5)
    model.train(False)
    model.eval()
    
    full_pred, full_lab = [], []
    
    TestLoss, TestAcc = 0.0, 0.0
    for data, target in testdata:
        if t_gpu:
            data, target = data.cuda(), target.cuda()

        output = model(data)
        loss = loss_fn(output, target)

        _, pred = torch.max(output.data, 1)
        TestLoss += loss.item() * data.size(0)
        TestAcc += torch.sum(pred == target.data)
        torch.cuda.empty_cache()
        full_pred += pred.tolist()
        full_lab += target.data.tolist()

    TestLoss = TestLoss / len(testdata.dataset)
    TestAcc = TestAcc / len(testdata.dataset)
    print(f'Loss: {TestLoss} Accuracy: {TestAcc}%')
    return full_pred, full_lab
    

testset = datasets.ImageFolder(test_dosya_dizini, 
                           transform=transforms.Compose([transforms.Resize(255),
                                                 transforms.CenterCrop(224),                                                              
                                                 transforms.ToTensor(),
                                                ]))
test_dl = DataLoader(testset, batch_size=32)

pred, lab = test_loop(model, test_dl, loss_fn, True)

from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

cm  = confusion_matrix(lab, pred)
plt.figure()
plot_confusion_matrix(cm,figsize=(12,8),cmap=plt.cm.Greens)
plt.xticks(range(2), ['Normal', 'Covid19'], fontsize=16)
plt.yticks(range(2), ['Normal', 'Covid19'], fontsize=16)
plt.xlabel('Tahmin edilen label',fontsize=18)
plt.ylabel('Doğru Label',fontsize=18)
plt.show()

tn, fp, fn, tp = cm.ravel()

accuracy = (np.array(pred) == np.array(lab)).sum() / len(pred)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = 2*((precision*recall)/(precision+recall))

print("Modelin doğruluğu (accuracy) {:.2f}".format(accuracy))
print("Modelin geri çağrılması (recall) {:.2f}".format(recall))
print("Modelin hassasiyeti {:.2f}".format(precision))
print("Modelin skoru {:.2f}".format(f1))

# Modeli kaydedelim

torch.save(model.state_dict(), 'kayit.pth')