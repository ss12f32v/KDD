from torch.utils.data import DataLoader
from torch.autograd import Variable
from Dataloader.Seq2Seq_DataLoader import DataTransformer
import pandas as pd 
import numpy as np 
import torch
import torch.nn as nn

from Model.Encoder import VanillaEncoder
from Model.Decoder import VanillaDecoder
from Model.Seq2Seq import Seq2Seq
from Trainer.Trainer import Trainer

from Tensorboard.Tensorboard import Logger

if __name__ == "__main__":


    data_transformer = DataTransformer(path = "Data/beijing/beijing_2017_1_2018_3_aq.csv",
                                    use_cuda = True)


    Enc = VanillaEncoder(hidden_size = 128,
                        output_size = 6).cuda()


    Dec = VanillaDecoder(input_size = 6,
                        hidden_size = 128, 
                        output_size = 6,
                        use_cuda = True).cuda()

    seq2seq = Seq2Seq(encoder=Enc,
                    decoder=Dec)

    train_logger = Logger('./logs/seq2seq_highway_train/emb100-h128-lr1e-3-batch32-ep25-2day-DECONLY')
    valid_logger = Logger('./logs/seq2seq_highway_valid/emb100-h128-lr1e-3-batch32-ep25-2day-DECONLY')
    loggers = (train_logger, valid_logger)
    trainer = Trainer(seq2seq, data_transformer, loggers= loggers, learning_rate = 0.001, use_cuda= True)

    
    trainer.train(num_epochs=25, 
                batch_size=32, 
                window_size = 2,
                pretrained=False,
                valid_portion = 0.8)
