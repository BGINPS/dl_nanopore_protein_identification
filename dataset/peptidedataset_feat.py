from utils import io
from utils.util import resize
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from sklearn.preprocessing import LabelEncoder
# data_name_list = ['hp1_1', 'hp1_2', 'hp1_3', 'hp1_4', 'hp1_5', 'hp1_6', 'hp1_7', 'hp1_8', 'hp1_9',
#                   'hp2_1', 'hp2_2', 'hp2_3', 'hp2_4', 'hp2_5', 'hp2_6', 'hp2_7']

# data_name_list = ['hp1_1', 'hp1_2', 'hp1_3', 'hp1_4', 'hp1_5', 'hp1_6', 'hp1_7', 'hp1_8', 'hp1_9',
#                   'hp2_1', 'hp2_2', 'hp2_3', 'hp2_4', 'hp2_5', 'hp2_6', 'hp2_7',
#                   'hp3_1', 'hp3_2', 'hp3_3', 'hp3_4', 'hp3_5', 'hp3_6', 'hp3_7', 'hp3_9', 'hp3_10', 'hp3_11', 'hp3_12'
#                  ]

## updated list
data_name_list = ['hp1_1', 'hp1_2', 'hp1_3', 'hp1_4', 'hp1_5', 'hp1_6', 'hp1_7', 'hp1_8', 'hp1_9',
                  'hp2_1', 'hp2_2', 'hp2_3', 'hp2_4', 'hp2_5', 'hp2_6',
                  'hp3_2', 'hp3_3', 'hp3_4', 'hp3_5', 'hp3_6', 'hp3_7', 'hp3_9', 'hp3_10', 'hp3_11'
                 ]

# data_name_list = ['bm1_14', 'bm1_16', 'bm1_17', 'bm1_26', 'bm1_28', 'bm1_29', 'bm1_36', 'bm1_38', 'bm1_42', 'bm1_7']

# data_name_list = ['hp1_1', 'hp1_2', 'hp1_3', 'hp1_4', 'hp1_5', 'hp1_6', 'hp1_7', 'hp1_8', 'hp1_9']

le = LabelEncoder()
le.fit(data_name_list)

class PeptideDataset_feature(Dataset):
    """
    The Peptide Dataset reads peptide signal from a hp dict
    """

    def __init__(self, hp_dict: dict , resize_length: int = 512, padding: int = 0):
        self.hp_dict = hp_dict
        self.resize_length = resize_length
        self.padding = padding
        self.keys_list = list( hp_dict.keys() )
        self.number_class = len(le.classes_)

        ## define the transform
        self.trans = transforms.Compose([       ### to do ---------update transforms, add crop, resize, noise, smooth...
            # transforms.CenterCrop(10),
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.hp_dict.keys())

    def __getitem__(self, i) -> (torch.Tensor, torch.Tensor):
        read_i = self.keys_list[i]

        read_i = self.hp_dict[ read_i ]

        ## cut the peptide signal and normalized by I0
        peptide_signal = read_i['signal'][read_i['window'][0]-self.padding : read_i['window'][1]+self.padding  ] / read_i['openpore']
        peptide_signal = resize( peptide_signal, self.resize_length )
        ### to do make the resize function into transforms
        # peptide_signal = self.trans(peptide_signal)
        peptide_signal = torch.tensor( peptide_signal, dtype=torch.float).unsqueeze(0)

        ## collect features: pd/rd
        feature = [read_i['pd2rd'], read_i['window_i2i0_mean'], read_i['window_i2i0_std'] ]
        peptide_feat = torch.tensor( feature, dtype=torch.float )

        label = le.transform( [read_i['label'] ] )
        label = torch.tensor(label[0], dtype=torch.long)
        label = torch.nn.functional.one_hot(label, self.number_class)

        return peptide_signal, peptide_feat, label.float()



class PeptideDataset_feat_infer(Dataset):
    """
    The dataset is only used to inference
    The Peptide Dataset reads peptide signal from a hp dict
    """

    def __init__(self, hp_dict: dict , resize_length: int = 512, padding: int = 0):
        self.hp_dict = hp_dict
        self.resize_length = resize_length
        self.padding = padding
        self.keys_list = list( hp_dict.keys() )
        self.number_class = len(le.classes_)

        ## define the transform
        self.trans = transforms.Compose([       ### to do ---------update transforms, add crop, resize, noise, smooth...
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.hp_dict.keys())

    def __getitem__(self, i) -> (torch.Tensor, torch.Tensor):
        read_i = self.keys_list[i]

        read_i = self.hp_dict[ read_i ]

        ## cut the peptide signal and normalized by I0
        peptide_signal = read_i['signal'][read_i['window'][0]-self.padding : read_i['window'][1]+self.padding  ] / read_i['openpore']
        peptide_signal = resize( peptide_signal, self.resize_length )
        ### to do make the resize function into transforms
        # peptide_signal = self.trans(peptide_signal)
        peptide_signal = torch.tensor( peptide_signal, dtype=torch.float).unsqueeze(0)

        ## collect features: pd/rd
        feature = [read_i['pd2rd'], read_i['window_i2i0_mean'], read_i['window_i2i0_std']]
        peptide_feat = torch.tensor(feature, dtype=torch.float)

        return peptide_signal, peptide_feat


## test
if __name__ =="__main__":
    tdir = '../data/cleaned/'
    hp_clean = io.read_pickle('../data/hp12.clean.pkl')

    dataset = PeptideDataset_feature(hp_clean)
    for i in range(100):
        signal, label =  dataset.__getitem__(i*300)
        print( signal.shape)
        print( label )

