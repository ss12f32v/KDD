from torch.utils.data import DataLoader
from torch.autograd import Variable
from Dataloader.Seq2Seq_DataLoader import DataTransformer
import pandas as pd 
import numpy as np 
import torch
import torch.nn as nn

from Model.Encoder import VanillaEncoder, BidirectionalGRUEncoder
from Model.Decoder import VanillaDecoder
from Model.Seq2Seq import Seq2Seq
from Trainer.Trainer import Trainer

from Tensorboard.Tensorboard import Logger

if __name__ == "__main__":


    data_transformer = DataTransformer(path="Data/beijing/beijing_2017_1_2018_3_aq.csv",
                                    use_cuda=True)


    Enc = BidirectionalGRUEncoder(hidden_size=512,
                                    output_size=3,
                                    embedding_size=100,
                                    num_layers=2).cuda()


    Dec = VanillaDecoder(input_size=3,
                        hidden_size=512, 
                        output_size=3,
                        use_cuda=True).cuda()

    seq2seq = Seq2Seq(encoder=Enc,
                    decoder=Dec)
    time_lag = 10
    train_logger = Logger('./logs/yuhua/no_embedding_absL-val0.8-emb100-h512-win10-lag' + str(time_lag) + '-EBI-LE1-lr1e-3-batch32-ep30-2day-DECONLY_train')
    valid_logger = Logger('./logs/yuhua/no_embedding_absL-val0.8-emb100-h512-win10-lag' + str(time_lag) + '-EBI-LE1-lr1e-3-batch32-ep30-2day-DECONLY_valid')
    loggers = (train_logger, valid_logger)
    trainer = Trainer(seq2seq, data_transformer, loggers=loggers, learning_rate=0.001, use_cuda=True)

    
    trainer.train(num_epochs=50, 
                batch_size=32, 
                window_size=10,
                pretrained=False,
                valid_portion=0.8,
                time_lag=time_lag)
