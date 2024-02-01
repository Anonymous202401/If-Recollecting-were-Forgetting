import os

import pandas as pd
from torch.utils.data import Dataset

from PIL import Image



def data_processing():
    df1 = pd.read_csv('./data/celeba/list_attr_celeba.txt', sep="\s+", skiprows=1, usecols=['Male'])
    df1.loc[df1['Male'] == -1, 'Male'] = 0

    df2 = pd.read_csv('./data/celeba/list_eval_partition.txt', sep="\s+", skiprows=0, header=None)
    df2.columns = ['Filename', 'Partition']
    df2 = df2.set_index('Filename')

    df3 = df1.merge(df2, left_index=True, right_index=True)
    # df3.head()


    # df3.to_csv('./data/celeba/celeba-gender-partitions.csv')
    # df4 = pd.read_csv('./data/celeba/celeba-gender-partitions.csv', index_col=0)
    # df4.head()


    # df4.loc[df4['Partition'] == 0].to_csv('./data/celeba/celeba-gender-train.csv')
    # df4.loc[df4['Partition'] == 1].to_csv('./data/celeba/celeba-gender-valid.csv')
    # df4.loc[df4['Partition'] == 2].to_csv('./data/celeba/celeba-gender-test.csv')
        # 指定训练集和测试集的具体数量
    train_samples = 4000  # 训练集样本数量
    test_samples = 400   # 测试集样本数量

    # 切片获取指定数量的训练集和测试集
    train_df = df3[df3['Partition'] == 0].head(train_samples)
    test_df = df3[df3['Partition'] == 2].head(test_samples)

    # 将剩余的数据划分为验证集
    valid_df = df3[~df3.index.isin(train_df.index) & ~df3.index.isin(test_df.index)]

    train_df.to_csv('./data/celeba/celeba-gender-train.csv')
    valid_df.to_csv('./data/celeba/celeba-gender-valid.csv')
    test_df.to_csv('./data/celeba/celeba-gender-test.csv')



class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, csv_path, img_dir, transform=None):
        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df.index.values
        self.y = df['Male'].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))

        if self.transform is not None:
            img = self.transform(img)

        label = self.y[index]
        return img, label

    def __len__(self):
        return self.y.shape[0]

if __name__ == '__main__':
    data_processing()