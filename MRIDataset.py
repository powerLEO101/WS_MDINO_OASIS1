import numpy as np
import pandas as pd
from os.path import join

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data.sampler import WeightedRandomSampler

from PIL import Image

def cdr_to_label(x): # 0: NonDemented 1: MildlyDemented 2: Demented
    if x != x:
        return 0
    x = float(x)
    from_cdr = {
        0: 0,
        0.5: 1,
        1: 2,
        2: 2
    }
    return from_cdr[x]

debug_array = {}

class MRIDataset(Dataset):
    
    def __init__(self, csv_path, root_path, ext, ws_class, transform=None, shuffle=True, ws=False, ignore=False):
        super().__init__()
        self.ws = ws
        self.df = pd.read_csv(csv_path)
        self.root_path = root_path
        self.ext = ext
        self.transform = transform
        self.rand_state = np.random.RandomState(233)
        self.given_index = 15
        self.samples = []
        self.samples_df = []
        self.ws_class = ws_class
        for idx in range(0, len(self.df)):
            age = int(self.df.iloc[idx]['Age'])
            if age < 62 and ignore: 
                continue
            file_name = self.df.iloc[idx]['ID'] + '_' + self.ext + '.npy'
            file_path = join(self.root_path, file_name)
            p_class = int(self.df.iloc[idx][ws_class])
            cdr = self.df.iloc[idx]['CDR']
            label = cdr_to_label(cdr)
            self.samples.append([p_class, file_path, label])

        if shuffle:
            self.rand_state.shuffle(self.samples)
        self.samples_df = pd.DataFrame(self.samples)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        p_class, file_path, label = self.samples[idx]
        data = np.load(file_path)

        data = Image.fromarray(data[self.given_index])
        data = self.transform(data)

        if self.ws:
            rand_image_pool = self.samples_df[self.samples_df[0] == p_class][1].values
            rand_image_3d = np.load(self.rand_state.choice(rand_image_pool))
            rand_image = Image.fromarray(rand_image_3d[self.given_index])
            rand_image = self.transform(rand_image)
            data = data[:2] + rand_image[2:]

        return data, label
    
def make_weights(images, nclasses):
    '''return weights, counts'''
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight, count


def train_test_split(df, random_state=233):
    df_split = []
    df_split.append(df[df['CDR'] == 0])
    df_split.append(df[df['CDR'] == 0.5])
    df_split.append(pd.concat([df[df['CDR'] == 1], df[df['CDR'] == 2]]))
    train = []; test = []
    for item in df_split:
        df_train = item.sample(frac=0.8, random_state=random_state)
        df_test = item.drop(df_train.index)
        train.append(df_train); test.append(df_test)
    train = pd.concat(train)
    test = pd.concat(test)
    return train, test

def create_class():
    import pandas as pd
    from sklearn.cluster import KMeans
    df_raw = pd.read_csv('/home/leo101/Work/Harp/Data/oasis1.csv')
    df = df_raw.drop(['Delay', 'M/F', 'Hand', 'Unnamed: 0', 'ASF', 'CDR'] , axis=1)
    df1 = df[df['MMSE'] == 0].drop(['Educ', 'SES', 'MMSE'], axis=1)
    df2 = df.drop(df1.index).drop(['eTIV', 'nWBV'], axis=1)

    model1 = KMeans(20, n_init='auto'); model2 = KMeans(20, n_init='auto')
    df1_pseudoclass = model1.fit_predict(df1.drop(['ID'], axis=1))
    df1['pseudo_class'] = df1_pseudoclass
    df2_pseudoclass = model2.fit_predict(df2.drop(['ID'], axis=1))
    df2['pseudo_class'] = df2_pseudoclass + 20

    df_raw['pseudo_class'] = df1['pseudo_class']
    df_raw['pseudo_class'].fillna(df2['pseudo_class'], inplace=True)
    df_raw.to_csv('/home/leo101/Work/Harp/Data/oasis2.csv')

if __name__ == '__main__':
    
    from torchvision import transforms
    dataset = MRIDataset('/home/leo101/Work/Harp/Data/OASIS1_mmse.csv', '/home/leo101/Work/Harp/Data/OASIS_NEW', 'y', ws_class='class_mmse', transform=transforms.ToTensor(), ws=False)
    #dataset.samples_df.to_csv('./test.csv')
    print([sample[0] for sample in dataset.samples])
    #print(len(np.unique(dataset.samples[:, 0])))