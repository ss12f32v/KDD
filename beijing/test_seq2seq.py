from torch.utils.data import DataLoader
from torch.autograd import Variable
from beijing.Dataloader.Seq2Seq_DataLoader import DataTransformer
import pandas as pd 
import numpy as np 
import torch
import pandas as pd
import torch.nn as nn

from beijing.Model.Encoder import VanillaEncoder, BidirectionalGRUEncoder
from beijing.Model.Decoder import VanillaDecoder
from beijing.Model.Seq2Seq import Seq2Seq
from beijing.eval.evaluator import Evaluator

from beijing.Tensorboard.Tensorboard import Logger

if __name__ == "__main__":

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

    seq2seq.load_state_dict(torch.load('Model_CheckPoint/seq2seqModel.pt'))
    
    evaluator = Evaluator()
    res = evaluator.evaluate(path='bj_for_encode.csv', model=seq2seq)
    df = pd.DataFrame()
    df['PM25'] = res[:, 0]
    df['PM10'] = res[:, 1]
    df['O3'] = res[:, 2]
    df.to_csv("BJ_2425", index=False)