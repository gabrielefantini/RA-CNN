
from torch.utils.data import Dataset, DataLoader
import cv2
import torchvision.transforms as transforms
import torch
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

class PlantDataset(Dataset):
        def __init__(self, df, image_dir="data\\train_images\\"):

            std = 1. / 255.
            means = [109.97 / 255., 127.34 / 255., 123.88 / 255.]

            self.image_id = df['image'].values
            self.labels = df.iloc[:, 1:].values
            self.image_dir = image_dir
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=means,
                    std=[std]*3)
            ])

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            image_id = self.image_id[idx]
            label = torch.tensor(self.labels[idx].astype('int8'), dtype=torch.float32)
            
            image_path = self.image_dir + image_id
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            image = self.transform(image)

            return image, label


def get_plant_loader():

    #read data from csv
    df_train = pd.read_csv(f'data/train.csv')
    #print(df_train.head())

    #label distribution
    train_count = df_train['labels'].value_counts()
    #print(train_count)

    #split labels
    df_train['labels'] = df_train['labels'].apply(lambda string: string.split(' '))
    #print(df_train.head(n=12))

    #make label in binary form
    train_df_list = list(df_train['labels'])
    mlb = MultiLabelBinarizer()
    trainx = pd.DataFrame(mlb.fit_transform(train_df_list), columns=mlb.classes_, index=df_train.index)
    #print(trainx.head(n=12))

    #concat label in binary form with image name
    train_data = pd.concat([df_train, trainx], axis=1).drop('labels', axis=1)

    train, remaining = train_test_split(train_data, train_size=0.8)
    test, validation = train_test_split(remaining, test_size=0.5)
    print(train_data)

    print(len(train),len(test), len(validation))

    return {
        "train": PlantDataset(train),
        "test": PlantDataset(test),
        "validation": PlantDataset(validation) 
        }