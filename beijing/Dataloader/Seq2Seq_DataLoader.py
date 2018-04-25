from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from torch.autograd import Variable
from sklearn import preprocessing


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
        self.data = self.raw_data.dropna(axis='index', how='any')
        # self.data = self.raw_data.fillna(0)
        self.feature_expand()
        self.data.set_index(['stationId'], inplace=True)
        self.station_id = self.data.index.unique().tolist()
        self.every_station_data = []
        for station in self.station_id:
            self.every_station_data.append(self.data.loc[station,:].iloc[10:10282,:].values)
    
    def feature_expand(self):
        day_time = pd.to_datetime(self.data['utc_time']).dt

        # Expand some time feature
        self.data['hour'] = day_time.hour
        self.data['day'] = day_time.day
        self.data['month'] = day_time.month
        #self.data['year'] = day_time.year
        self.data['dayofweek'] = day_time.dayofweek
        self.data['dayofyear'] = day_time.dayofyear
        #self.data['is_weekend'] = (self.data['dayofweek']==5) | (self.data['dayofweek']==6)
        #self.data['work_time'] = ( self.data.hour > 6) & ( self.data.hour < 20)

        # Drop utc-time column. We already get some feature above
        self.data.drop(['utc_time'], axis=1, inplace = True)
        
        le = preprocessing.LabelEncoder()
        self.one_hot_candidate_columns = ['hour', 'day',
                                          'month',
                                          'dayofweek', 'dayofyear'
                                          ]
        # self.year_dict = {'2017':0, '2018':1}
        # self.data['year'] = self.data['year'].apply(lambda v: self.year_dict[v])

    def create_mini_batch(self, data, batch_size= 10, window_size=2, use_cuda=False, time_lag=10, shuffle_input=True):
        batch_list = []
        label_list = []
        
        # NOTE THAT SHOULD DO SPECIAL PROCESS FOR FINAL BATCH!!!
        for k in range(0, len(data)):     # 0, data_length, stride
            if k + window_size *24 + 48 + time_lag <= len(data):
                batch_list.append(data[k: k + window_size *24])
                label_list.append(data[k + window_size *24 + time_lag: k + window_size *24 + 48 + time_lag, 0:6])

        assert len(batch_list) == len(label_list)
        mini_batches = [
                np.array(batch_list[k: k + batch_size])
                for k in range(0, len(batch_list), batch_size)
        ]

        mini_labels = [
                np.array(label_list[k: k + batch_size])
                for k in range(0, len(label_list), batch_size)
        ]

        if shuffle_input:
            mini_batches, mini_labels = shuffle(mini_batches, mini_labels, random_state=42)
        return mini_batches, mini_labels
    
    def prepare_all_station_data(self, all_station_data, training_portion, batch_size, valid_batch_size, window_size, time_lag, shuffle_input):
        all_station_train = []
        all_station_valid = []

        all_station_train_input = []
        all_station_train_label = []

        all_station_valid_input = []
        all_station_valid_label = []

        for data in all_station_data:
            all_station_train.append(data[: int(len(data) * training_portion)])
            all_station_valid.append(data[ int(len(data) * training_portion):])
        
        for t in all_station_train:
            ci, cl = self.create_mini_batch(data=t, batch_size=batch_size, window_size=window_size, time_lag=time_lag, shuffle_input=shuffle_input)
            all_station_train_input += ci
            all_station_train_label += cl

        for t in all_station_valid:
            ci, cl = self.create_mini_batch(data=t, batch_size=valid_batch_size, window_size=window_size, time_lag=time_lag, shuffle_input=shuffle_input)
            all_station_valid_input += ci
            all_station_valid_label += cl

        if shuffle_input:
            all_station_train_input, all_station_train_label = shuffle(all_station_train_input, all_station_train_label, random_state=42)
        
        return all_station_train_input, all_station_train_label, all_station_valid_input, all_station_valid_label

    def variables_generator(self, batches, labels):
        for batch, label in zip (batches, labels):
            input_var = Variable(torch.FloatTensor(batch)).transpose(0, 1)  # time * batch
            target_var = Variable(torch.FloatTensor(label)).transpose(0, 1)  # time * batch

            if self.use_cuda:
                    input_var = input_var.cuda()
                    target_var = target_var.cuda()
            
            yield input_var, target_var



class TestDataTransformer(object):

    def __init__(self, london, path , use_cuda):
        self.use_cuda = use_cuda
        self.path = path
        self.columns = ['PM2.5','PM10' ,'O3']
        self.data = pd.read_csv(self.path)
        self.london = london
        self.prepare_data()

    def prepare_data(self):
        self.data = self.data.fillna(0)
        #self.feature_expand()
        self.data.set_index(['station_id'], inplace=True)
        if self.london:
            self.station_id= ['BL0', 'CD9', 'CD1', 'GN0', 
                              'GR4', 'GN3', 'GR9', 'HV1',
                              'KF1', 'LW2', 'ST5', 'TH4', 'MY7' ]
        else:
            self.station_id = self.data.index.unique().tolist()
        self.every_station_data = []
        self.data = self.data.drop(['id', 'time'], axis=1)

        for station in self.station_id:
            self.every_station_data.append(self.data.loc[station,:].values)
        for i in range(len(self.every_station_data)):
            self.every_station_data[i] = self.every_station_data[i][-240:]
            

    def create_mini_batch(self, data, use_cuda=False):
        mini_batches = np.array([data])
        # print(mini_batches.shape)
        return mini_batches
    
    def variables_generator(self, batch):
        input_var = Variable(torch.FloatTensor(batch)).transpose(0, 1)  # time * batch
        if self.use_cuda:
            input_var = input_var.cuda()
        return input_var