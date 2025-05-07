from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
import numpy as np
import pandas as pd
import torch


class BaseDataset(Dataset):
    def __init__(self, df, normalize=False, signal_length=250):

        self.df = df
        if normalize:
            self.input_scaler = MinMaxScaler()
        start_index = self.df.columns.get_loc('0.0')
        length_of_signal = signal_length
        end_index = start_index + length_of_signal
        self.latency_values = self.df.iloc[:, start_index:end_index]

        self.latency_values = [float(value) for value in self.latency_values]
        self.x = self.df.iloc[:, start_index:end_index].values
        if normalize:
            self.x = self.input_scaler.fit_transform(self.x)

        self.unique_ids = self.df['unique_id'].values
        self.additional_info = self.df[[
            'id', 'stimulus side', 'intensity', 'loss type - right', 'loss type - left']]

    def __len__(self):
        return len(self.df)

    def get_additional_info(self, unique_id):
        if isinstance(unique_id, torch.Tensor):
            unique_id = unique_id.cpu().numpy()
        if isinstance(unique_id, np.ndarray):
            records = self.df[self.df['unique_id'].isin(unique_id)]
        else:
            records = self.df.loc[self.df['unique_id'] == unique_id]
        return records

    def create_pairs(self):
        pairs = []
        grouped = self.df.groupby(['id', 'stimulus side'])
        for (id, side), group in grouped:
            highest_intensity_idx = group['intensity'].idxmax()
            highest_intensity_signal = self.x[highest_intensity_idx]
            for idx in group.index:
                signal = self.x[idx]
                pairs.append((highest_intensity_signal, signal,
                              self.y[idx], self.unique_ids[idx]))
        return pairs


class DetectorDataset(BaseDataset):
    def __init__(self, df, normalize_y=False, single_mode=False):
        super().__init__(df)
        self.single_mode = single_mode
        self.y = df['mapped_targets']
        if normalize_y:
            self.y = self.y / 250

        self.encoded_y = self.create_encoded_y()
        if not single_mode:
            self.pairs = self.create_pairs()

    def create_encoded_y(self):
        encoded_y = []
        for value in self.y:
            if np.isnan(value):
                # Create a vector with 0 for existence and None for location
                y_vector = [0, 0]
            else:
                # Create a vector with 1 for existence and the location value
                y_vector = [1, value]
            encoded_y.append(y_vector)
        return np.array(encoded_y, dtype=np.float32)

    def create_pairs(self):
        pairs = []
        grouped = self.df.groupby(['id', 'stimulus side'])
        for (id, side), group in grouped:
            highest_intensity_idx = group['intensity'].idxmax()
            highest_intensity_signal = self.x[highest_intensity_idx]
            for idx in group.index:
                signal = self.x[idx]
                pairs.append((highest_intensity_signal, signal,
                              self.encoded_y[idx], self.unique_ids[idx]))
        return pairs

    def __getitem__(self, idx):
        if self.single_mode:
            x = torch.tensor(self.x[idx], dtype=torch.float32)
            y = torch.tensor(self.encoded_y[idx], dtype=torch.float32)
            unique_id = self.unique_ids[idx]
            return x, y, unique_id
        else:
            signal1, signal2, y, unique_id = self.pairs[idx]
            signal1 = torch.tensor(signal1, dtype=torch.float32)
            signal2 = torch.tensor(signal2, dtype=torch.float32)
            combined_signal = torch.stack((signal1, signal2), dim=0)
            y = torch.tensor(y, dtype=torch.float32)
            return combined_signal, y, unique_id
