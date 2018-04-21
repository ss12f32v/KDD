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


    enc = HighwayEncoder(hidden_size=256,
                        output_size=6).cuda()


    dec = VanillaDecoder(input_size=6,
                        hidden_size=256, 
                        output_size=6,
                        use_cuda=True).cuda()

    seq2seq = Seq2Seq(encoder=enc,
                    decoder=dec)

    train_logger = Logger('./logs/seq2seq_highway_train/emb100-h256-lr1e-3-L7-batch32-ep30-2day-DECONLY')
    valid_logger = Logger('./logs/seq2seq_highway_valid/emb100-h256-lr1e-3-L7-batch32-ep30-2day-DECONLY')
    loggers = (train_logger, valid_logger)
    trainer = Trainer(seq2seq, data_transformer, loggers= loggers, learning_rate = 0.001, use_cuda= True)

    
    trainer.train(num_epochs=30, 
                batch_size=32, 
                window_size = 2,
                pretrained=False,
                valid_portion=0.8)
