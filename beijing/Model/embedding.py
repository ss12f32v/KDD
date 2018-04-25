import torch
import torch.nn as nn
from beijing.Dataloader.data_config import feature_config

class FeatureEmbedding(nn.Module):

    def __init__(self, embedding_size):
        super(FeatureEmbedding, self).__init__()

        self.embedding_size = embedding_size

        self.hour_embedding = nn.Embedding(25, 8)
        self.day_embedding = nn.Embedding(32, 10)
        self.month_embedding = nn.Embedding(13, 4)
        #self.year_embedding = nn.Embedding(2, embedding_size)
        self.dayofweek_embedding = nn.Embedding(8, 2)
        self.dayofyear_embedding = nn.Embedding(367, 90)
        #self.is_weekend_embedding = nn.Embedding(2, embedding_size)
       # self.work_time_embedding = nn.Embedding(2, embedding_size)
        self.embedding_size = 8 + 10 + 4 + 2 + 90

    def forward(self, input_seqs):
        hour_embedded = self.hour_embedding(input_seqs[:, :, feature_config['hour']].long())
        day_embedded = self.day_embedding(input_seqs[:, :, feature_config['day']].long())
        month_embedded = self.month_embedding(input_seqs[:, :, feature_config['month']].long())
        #year_embedded = self.year_embedding(input_seqs[:, :, feature_config['year']].long())
        dayofweek_embedded = self.dayofweek_embedding(input_seqs[:, :, feature_config['dayofweek']].long())
        dayofyear_embedded = self.dayofyear_embedding(input_seqs[:, :, feature_config['dayofyear']].long())
        #is_weekend_embedded = self.is_weekend_embedding(input_seqs[:, :, feature_config['is_weekend']].long())
        #work_time_embedded = self.work_time_embedding(input_seqs[:, :, feature_config['work_time']].long())

        feature_embedding = torch.cat((hour_embedded,
                                        day_embedded,
                                        month_embedded,
                                        #year_embedded,
                                        dayofweek_embedded,
                                        dayofyear_embedded,
                                        #is_weekend_embedded,
                                        #work_time_embedded
                                        ), dim= 2)
        return feature_embedding