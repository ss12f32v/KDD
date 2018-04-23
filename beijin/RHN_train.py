from torch.utils.data import DataLoader
from torch.autograd import Variable
from Dataloader.Seq2Seq_DataLoader import DataTransformer
import pandas as pd 
import numpy as np 
import torch
import torch.nn as nn

from Model.Encoder import HighwayEncoder
from Model.Decoder import VanillaDecoder
from Model.Seq2Seq import Seq2Seq
from Trainer.Trainer import Trainer

from Tensorboard.Tensorboard import Logger

if __name__ == "__main__":


    data_transformer = DataTransformer(path = "Data/beijing/beijing_2017_1_2018_3_aq.csv",
                                    use_cuda = True)


    enc = HighwayEncoder(hidden_size=512,
                        output_size=6,
                        embedding_size=100,
                        depth=3).cuda()


    dec = VanillaDecoder(input_size=6,
                        hidden_size=512, 
                        output_size=6,
                        use_cuda=True).cuda()

    seq2seq = Seq2Seq(encoder=enc,
                    decoder=dec)

    train_logger = Logger('./logs/seq2seq_highway_train/emb100-he512-d300-lr1e-3-window10-L3-batch32-ep100-2day-DECONLY-NoDrop')
    valid_logger = Logger('./logs/seq2seq_highway_valid/emb100-he512-d300-lr1e-3-window10-L3-batch32-ep100-2day-DECONLY-NoDrop')
    loggers = (train_logger, valid_logger)
    trainer = Trainer(seq2seq, data_transformer, loggers=loggers, learning_rate=1e-3, use_cuda=True)

    
    trainer.train(num_epochs=3700, 
                batch_size=32, 
                window_size = 10,
                pretrained=False,
                valid_portion=0.8,
                time_lag=10)
