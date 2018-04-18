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


if __name__ == "__main__":


    data_transformer = DataTransformer(path = "Data/beijing/beijing_2017_1_2018_3_aq.csv",
                                    use_cuda = True)


    Enc = VanillaEncoder(hidden_size = 300,
                        output_size = 6).cuda()


    Dec = VanillaDecoder(input_size = 6,
                        hidden_size = 300, 
                        output_size = 6,
                        use_cuda = True).cuda()

    seq2seq = Seq2Seq(encoder=Enc,
                    decoder=Dec)

    trainer = Trainer(seq2seq, data_transformer, learning_rate = 0.01, use_cuda= True)


    trainer.train(num_epochs=1, 
                batch_size=10, 
                window_size = 2,
                pretrained=False)
