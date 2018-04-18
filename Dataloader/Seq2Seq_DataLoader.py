from torch.utils.data import DataLoader
from torch.autograd import Variable

import pandas as pd 
import numpy as np 
import torch


class DataTransformer(object):
    def __init__(self, path , use_cuda):
        self.use_cuda = use_cuda
        self.path = path
        # self.path = "Data/beijing/beijing_2017_1_2018_3_aq.csv"
        self.columns = ['PM2.5','PM10','NO2','CO','O3','SO2']
        self.raw_data = pd.read_csv(self.path)
        self.prepare_data()

    def prepare_data(self):
        self.data = self.raw_data.fillna(0)
        self.data.set_index(['stationId'], inplace=True)
        self.station_id = self.data.index.unique().tolist()
        self.every_station_data = []
        for station in self.station_id:
            self.every_station_data.append(self.data.loc[station,:].iloc[10:10282,:].drop(['utc_time'], axis=1).as_matrix())

    def mini_batch_generator(self, data, batch_size= 10, window_size=10, use_cuda= False ):
        one_pair_length = window_size * 24  + 48 
        number_of_pair_data = int( (len(data)-48) / window_size * 24)
            
        batch_list = []
        label_list = []
        for k in range(0, len(data), window_size *24):    
            if k + window_size *24 + 48 <= len(data):
                batch_list.append(data[k: k + window_size *24])
        
        for k in range(0, len(data), window_size *24):
            if k + window_size *24 + 48 <= len(data):
                label_list.append(data[k + window_size *24: k + window_size *24 +48])

        
        assert len(batch_list) == len(label_list)
        
        mini_batches = [
                np.array(batch_list[k: k + batch_size])
                for k in range(0, len(batch_list), batch_size)
            ]
        mini_labels = [
                np.array(label_list[k: k + batch_size])
                for k in range(0, len(label_list), batch_size)
            ]
        
        for batch,label in zip (mini_batches,mini_labels):
            # assert batch.shape == label.shape
            input_var = Variable(torch.FloatTensor(batch)).transpose(0, 1)  # time * batch
            target_var = Variable(torch.FloatTensor(label)).transpose(0, 1)  # time * batch
            # print(input_var.shape)
            # print(target_var.shape)

            if self.use_cuda:
                    input_var = input_var.cuda()
                    target_var = target_var.cuda()
            
            yield input_var, target_var
    
if __name__ == '__main__':
    data_tran = DataTransformer(path = "../Data/beijing/beijing_2017_1_2018_3_aq.csv",
                                use_cuda = True)

    for station_data in data_tran.every_station_data:
        for i , (batch, label) in enumerate(data_tran.mini_batch_generator(station_data)):
            print(batch.shape)
            print(label.shape)
            print()