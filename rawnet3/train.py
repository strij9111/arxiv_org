import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import torchaudio

import argparse
import os
import numpy as np
from tensorboardX import SummaryWriter

from model.RawNet3 import MainModel
from classifier.aamsoftmax import aamsoftmax
#from module.dataset import VoxDataset

from ptUtils.hparams import HParam
from ptUtils.writer import MyWriter

from module.common import run


if torch.cuda.is_available():
    print("GPU(s) available: ", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print("GPU ", i, ":", torch.cuda.get_device_name(i))
else:
    print("No GPU available.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--default', type=str, required=True,
                        help="default configuration")

    parser.add_argument('--version_name', '-v', type=str, required=True,
                        help="version of current training")
    parser.add_argument('--chkpt', type=str, required=False, default=None)
    parser.add_argument('--step', '-s', type=int, required=False, default=0)
    parser.add_argument('--device', '-d', type=str,
                        required=False, default="cuda:0")
    args = parser.parse_args()

    hp = HParam(args.config, args.default)
    print("NOTE::Loading configuration : "+args.config)

    device = args.device
    version = args.version_name
#    torch.cuda.set_device(device)

    batch_size = 110
    num_epochs = 20
    num_workers = 1

    best_loss = 1e7

    # load

    modelsave_path = './'+'chkpt' + '/' + version
    log_dir = './'+'log'+'/'+version

    os.makedirs(modelsave_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

#    writer = MyWriter(hp, log_dir)

    # target
#    train_dataset = VoxDataset(hp, is_vox1=False)
    train_dataset = torchaudio.datasets.SPEECHCOMMANDS(
        "e:\\StreamingASR\\train\\", subset="training", download=False)
#    train_dataset = WAVDataset(path=["train2\\movix1", "train2\\other"])
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = MainModel().to(torch.device('cuda'))
    classifier = aamsoftmax(hp).to(device)

#    print('NOTE::Loading pre-trained model')
#    model.load_state_dict(torch.load("e:\\StreamingASR\\ACS\\rawnet3\\best.pt", map_location=device))

    criterion = None
    if hp.loss.type == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    else:
        raise Exception("ERROR::Unknown loss : {}".format(hp.loss.type))

    optimizer = torch.optim.Adam(model.parameters(), lr=hp.train.adam)
    scaler = torch.cuda.amp.GradScaler()

    if hp.scheduler.type == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=hp.scheduler.CosineAnnealingLR.T_max, eta_min=hp.scheduler.CosineAnnealingLR.eta_min)
    else:
        raise Exception("Unsupported sceduler type")

    step = args.step

    for epoch in range(num_epochs):
        ### TRAIN ####
        model.train()
        train_loss = 0
        for i, batch_data in enumerate(train_loader):
            data = batch_data[0]
            data = torch.squeeze(data)
            temp_list = list(batch_data[2])
            label = []
            for s in temp_list:
                if s == 'follow':
                    label.append(1)
                else:
                    label.append(0)
            label = torch.tensor(label)
            step += 1
            data = data.to(device)
            label = label.to(device)

            loss = run(data, label, model, classifier, criterion)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            print('TRAIN::{} : Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                version, epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

#            if step %  hp.train.summary_interval == 0:
#                writer.log_value(loss,step,'train loss : '+hp.loss.type)

        train_loss = train_loss/len(train_loader)
        torch.save(model.state_dict(), str(modelsave_path) +
                   '/epoch_{}_loss_{}.pt'.format(epoch, train_loss))
        scheduler.step()
