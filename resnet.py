import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

from helper_code import find_records, get_age, get_sampling_frequency, get_sex, load_header, get_label, load_signals

def get_dataset(records_list: list, data_folder: Path):
    n_recors = len(records_list)

    features = []
    labels = []
    for idx in tqdm(range(20), desc='Loading data'):
        record_path = (data_folder / records_list[idx]).absolute().__str__()
        header = load_header(record_path)
        signals, fields = load_signals(record_path)
        frequency = get_sampling_frequency(header)
        label = get_label(header)
        age = get_age(header)
        sex = get_sex(header)

        one_hot_encoding_sex = np.zeros(3, dtype=bool)
        if sex == 'Female':
            one_hot_encoding_sex[0] = 1
        elif sex == 'Male':
            one_hot_encoding_sex[1] = 1
        else:
            one_hot_encoding_sex[2] = 1
        
        features.append(signals)
        labels.append(label)

    return features, labels

class ECGDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signals = self.features[idx]
        label = self.labels[idx]

        return self.to_tensor(signals), label

    def to_tensor(self, signals):
        return torch.tensor(signals, dtype=torch.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a ResNet model for ECG classification of Chagas disease')
    parser.add_argument('--train_data', type=str, help='Path to the training data csv file')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    args = parser.parse_args()


    train_data_folder = Path(args.train_data)
    train_records = find_records(train_data_folder.absolute().__str__())
    
    data = get_dataset(records_list=train_records, data_folder=train_data_folder)

    X, y = data[0], data[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=args.seed)

    train_dataset = ECGDataset(X_train, y_train)
    train_dloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)

    print(train_dloader.dataset[0])