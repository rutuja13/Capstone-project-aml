import os
import math
import hashlib
import skimage.io
import numpy as np
import pandas as pd
from collections import defaultdict
from skimage.transform import resize

import torch
from torch.utils.data import Dataset

from data import CLASSES


class OrthonetBaseDataset(Dataset):
    def __init__(self, train_val_test, csv_path, data_path, transforms, n_folds=5, fold=1):
        self.train_val_test = train_val_test
        self.csv_path = csv_path
        self.data_path = data_path
        self.transforms = transforms
        self.n_folds = n_folds
        self.fold = fold

        self.samples = self.get_samples()
        self.summary()

    def __len__(self):
        return len(self.samples)

    def summary(self):
        print(f"Found {len(self.samples)} {self.train_val_test} samples from {self.csv_path}\n")

    def get_samples(self):
        raise NotImplementedError()

    def stats(self):
        samples_by_class = defaultdict(list)
        patients = []
        for sample in self.samples:
            samples_by_class[sample['labels']].append(sample['filenames'])
            patients.append(sample['patient_id'])
        print(f"{len(set(patients))} unique patients")
        print(f"{'Class':<50}Number of samples")
        for label, samples in samples_by_class.items():
            print(f"{label:<50}{len(samples)}")
        print("\n")

    def get_class_weights(self):
        n_samples_by_class = {cls: 0 for cls in CLASSES}
        for sample in self.samples:
            n_samples_by_class[sample['labels']] += 1
        median_count = np.median(list(n_samples_by_class.values()))
        class_weights = torch.tensor([math.sqrt(median_count/n) for n in n_samples_by_class.values()])
        return class_weights

    @staticmethod
    def get_train_val_for_patient_id(patient_id, fold, n_folds):
        rand_num = int(hashlib.md5(str.encode(str(patient_id))).hexdigest(), 16) / 16 ** 32
        val_fold = math.floor(rand_num * n_folds) + 1
        if val_fold == fold:
            return 'val'
        else:
            return 'train'


class OrthonetClassificationDataset(OrthonetBaseDataset):

    def get_samples(self):
        """Returns a list of dictionaries, of len(n_samples) and format:
            {
                'filenames': '0003_35_25_1_A-P02_UNIL.png',
                'labels': 'Knee_SmithAndNephew_Legion2',
                'patient_id': 3,
                'masks': '0003_35_25_1_A-P02_UNIL.png' <- only for TRAIN/VAL, only relevant for Unet
            }"""

        df = pd.read_csv(self.csv_path)

        if self.train_val_test == 'test':
            # Use all if testing
            pass
        else:
            patient_ids_all = sorted(list(df.patient_id.unique()))
            if self.train_val_test == 'test':  # We use all cases when testing
                patient_ids_selected = patient_ids_all
            else:
                patient_ids_selected = [pid for pid in patient_ids_all if self.get_train_val_for_patient_id(pid, self.fold, self.n_folds) == self.train_val_test]
            df = df[df['patient_id'].isin(patient_ids_selected)]
        return df.to_dict('records')

    def __getitem__(self, idx):
        sample = self.samples[idx]
        filepath = os.path.join(self.data_path, sample['filenames'])
        img = skimage.io.imread(filepath)
        label_name = sample['labels']
        label_id = CLASSES.index(label_name)

        img = self.transforms(image = img)['image'] / 255.

        x = torch.from_numpy(img).float()
        if len(x.shape) == 3:
            x.permute((2, 0, 1))  # Channels last -> first
        else:
            x.unsqueeze_(0)

        y = torch.tensor(label_id).long()

        return x, y, filepath, label_name


class OrthonetSegmentationDataset(OrthonetBaseDataset):
    def get_samples(self):
        """Returns a list of dictionaries, of len(n_samples) and format:
            {
                'filenames': '0003_35_25_1_A-P02_UNIL.png',
                'labels': 'Knee_SmithAndNephew_Legion2',
                'masks': '0003_35_25_1_A-P02_MASK.png',
                'patient_id': 3
            }"""

        df = pd.read_csv(self.csv_path)

        if self.train_val_test == 'test':
            # Use all if testing, regardless of whether mask is present
            pass
        else:
            df = df.dropna()
            df = df.loc[df['valid_mask'] == True].copy()
            df = df.loc[df['masks'] != ""].copy()  # Remove entries with missing masks
            patient_ids_all = sorted(list(df.patient_id.unique()))
            if self.train_val_test == 'test':  # We use all cases when testing
                patient_ids_selected = patient_ids_all
            else:
                patient_ids_selected = [pid for pid in patient_ids_all if self.get_train_val_for_patient_id(pid, self.fold, self.n_folds) == self.train_val_test]
            df = df[df['patient_id'].isin(patient_ids_selected)]
        return df.to_dict('records')

    def __getitem__(self, idx):
        sample = self.samples[idx]
        filepath = os.path.join(self.data_path, sample['filenames'])
        try:
            mask_path = os.path.join(self.data_path, sample['masks'])
        except Exception as e:
            print(f"Error finding {sample['masks']}")
            raise ValueError(f"Error finding {sample['masks']}: {e}")
        label_name = sample['labels']

        img = skimage.io.imread(filepath)
        mask = skimage.io.imread(mask_path) / 255.

        if img.shape[-2:] != mask.shape[-2:]:
            # print(f"Warning, mask mismatch for {os.path.basename(filepath)}")
            mask = resize(mask, img.shape[-2:], anti_aliasing=False)

        aug = self.transforms(image=img, mask=mask)

        img, mask = aug['image'] / 255., aug['mask']

        x = torch.from_numpy(img).float()
        if len(x.shape) == 3:
            x.permute((2, 0, 1))  # Channels last -> first
        else:
            x.unsqueeze_(0)

        y = torch.tensor(mask).float().unsqueeze(0)  # Float & un-squeeze as BCE loss

        return x, y, filepath, label_name
