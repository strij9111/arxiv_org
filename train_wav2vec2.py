"""
Обучение модели wav2vec2
"""
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from pytorch_metric_learning import losses

torch.cuda.empty_cache()
bundle = torchaudio.pipelines.WAV2VEC2_BASE
wav2vec2 = bundle.get_model()
wav2vec2.encoder.transformer.layers = wav2vec2.encoder.transformer.layers[:-4]

root_dir = "e:\\tf20\\data\\train"
batch_size = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CommandDataset(Dataset):

    def __init__(self, meta, root_dir, sample_rate, labelmap):
        self.meta = meta
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.labelmap = labelmap

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.meta['path'].iloc[idx]
        waveform, sample_rate = torchaudio.load(file_name)

        #        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)#[:, :10**5]
        waveform = torch.nn.functional.pad(waveform, (16000 - waveform.shape[1], 0))[0]

        label = self.meta['label'].iloc[idx]

        return waveform, self.labelmap[label]


labels = {
    'keyword': 1,
    'other': 0,
}

data = pd.DataFrame([
    {'label': i[0].split("\\")[-1], 'path': i[0] + "\\" + j}
    for i in os.walk(root_dir)
    for j in i[2]
])

# print(data.label.value_counts())
train, val, _, _ = train_test_split(data, data['label'], test_size=0.1)

train_dataset = CommandDataset(meta=train, root_dir=root_dir, sample_rate=bundle.sample_rate, labelmap=labels)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

val_dataset = CommandDataset(meta=val, root_dir=root_dir, sample_rate=bundle.sample_rate, labelmap=labels)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


class CommandClassifier(nn.Module):
    def __init__(self, feature_extractor):
        super(CommandClassifier, self).__init__()
        self.feature_extractor = feature_extractor
        self.linear = nn.Linear(768, len(labels))

    def forward(self, X):
        features = self.get_embeddings(X)
        logits = self.linear(features)
        return logits

    def get_embeddings(self, X):
        embeddings = self.feature_extractor(X)[0].mean(axis=1)
        return nn.functional.normalize(embeddings)


model = CommandClassifier(wav2vec2)
# model.load_state_dict(torch.load('model.pth'))
model.to(device)

EPOCHS = 10
lr = 0.00001

optimizer = optim.AdamW(model.parameters(), lr)
criterion = losses.ArcFaceLoss(len(labels), 768).to(device)

if __name__ == '__main__':
    for epoch in range(EPOCHS):

        model.train()

        train_loss = []
        for batch, targets in tqdm(train_dataloader, desc=f"Epoch: {epoch}"):
            if batch.shape[0] < 4:
                continue
            optimizer.zero_grad()
            batch = batch.to(device)
            targets = targets.to(device)

            predictions = model.get_embeddings(batch.squeeze())

            loss = criterion(predictions, targets)
            loss.backward()

            optimizer.step()

            train_loss.append(loss.item())

        print('Training loss:', np.mean(train_loss))

        model.eval()

        val_loss = []
        for batch, targets in tqdm(val_dataloader, desc=f"Epoch: {epoch}"):
            with torch.no_grad():
                batch = batch.to(device)
                targets = targets.to(device)

                predictions = model.get_embeddings(batch)

                loss = criterion(predictions, targets)

                val_loss.append(loss.item())

        print('Val loss:', np.mean(val_loss))

    torch.save(model.state_dict(), 'model.pth')