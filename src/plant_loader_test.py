import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
import torchvision.transforms as transforms
import torch
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split


class PlantDataset(Dataset):
    def __init__(self, df, image_dir="data\\test_images\\"):

        # Imagenet
        # std = [1. / 255., 1. / 255., 1. / 255.]
        # means = [109.97 / 255., 127.34 / 255., 123.88 / 255.]

        # calculated on this dataset
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        self.image_id = df['image'].values
        self.labels = df.iloc[:, 1:].values
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std)
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_id = self.image_id[idx]
        label = torch.tensor(self.labels[idx].astype(
            'int8'), dtype=torch.float32)

        image_path = self.image_dir + image_id
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transform(image)

        return image, label

    @staticmethod
    def tensor_to_img(x, imtype=np.uint8):
        '''
        # calculated on this dataset
        mean = [0.4874, 0.6265, 0.4082]
        std = [0.1634, 0.1441, 0.1704]
        '''
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        if not isinstance(x, np.ndarray):
            if isinstance(x, torch.Tensor):  # get the data from a variable
                image_tensor = x.data
            else:
                return x
            image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
            if image_numpy.shape[0] == 1:  # grayscale to RGB
                image_numpy = np.tile(image_numpy, (3, 1, 1))
            for i in range(len(mean)):
                image_numpy[i] = image_numpy[i] * std[i] + mean[i]
            image_numpy = image_numpy * 255
            # post-processing: tranpose and scaling
            image_numpy = np.transpose(image_numpy, (1, 2, 0))
        else:  # if it is a numpy array, do nothing
            image_numpy = x
        return image_numpy.astype(imtype)


def get_plant_loader():
    
    # read data from csv
    df_train = pd.read_csv(f'data/test.csv')
    print(df_train.head())

    # label distribution
    train_count = df_train['labels'].value_counts()
   # print(train_count)

    # split labels
    df_train['labels'] = df_train['labels'].apply(
        lambda string: string.split(' '))
    # print(df_train.head(n=12))

    # make label in binary form
    train_df_list = list(df_train['labels'])
    mlb = MultiLabelBinarizer()
    trainx = pd.DataFrame(mlb.fit_transform(train_df_list),
                          columns=mlb.classes_, index=df_train.index)
    print(trainx.head(n=12))

    # concat label in binary form with image name
    test_data = pd.concat([df_train, trainx], axis=1).drop('labels', axis=1)

    return PlantDataset(test_data)
