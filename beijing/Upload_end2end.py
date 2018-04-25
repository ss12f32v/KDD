

from torch.utils.data import DataLoader
from torch.autograd import Variable
from beijing.Dataloader.Seq2Seq_DataLoader import DataTransformer
import requests
import argparse
import pandas as pd 
import numpy as np 
import torch
import pandas as pd
import torch.nn as nn

from beijing.Model.Encoder import VanillaEncoder, BidirectionalGRUEncoder
from beijing.Model.Decoder import VanillaDecoder
from beijing.Model.Seq2Seq import Seq2Seq
from beijing.eval.evaluator import Evaluator

class bj_upload():

    def __init__(self):
        pass
    def get_data(self, bj_time_interval, ld_time_interval):
        bj_url = 'https://biendata.com/competition/airquality/bj/' + bj_time_interval +'/2k0d1d8'
        print(bj_url)
        respones= requests.get(bj_url)
        with open ("realtime_rawdata/bj_real_time.csv",'w') as f:
            f.write(respones.text)
            f.close()
        data = pd.read_csv("realtime_rawdata/bj_real_time.csv").drop(['NO2_Concentration', 'CO_Concentration', 'SO2_Concentration'], axis=1)
        data.to_csv("realtime_rawdata/bj_real_time.csv")
        
        ld_url = 'https://biendata.com/competition/airquality/ld/' + bj_time_interval +'/2k0d1d8'
        respones= requests.get(ld_url)
        with open ("realtime_rawdata/ld_real_time.csv",'w') as f:
            f.write(respones.text)
            f.close()
        data = pd.read_csv("realtime_rawdata/ld_real_time.csv").drop(['NO2_Concentration', 'CO_Concentration', 'SO2_Concentration'], axis=1)
        data.to_csv("realtime_rawdata/ld_real_time.csv")

    def bjForward(self, bj_model_pt):   
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

        seq2seq.load_state_dict(torch.load(bj_model_pt))
        
        evaluator = Evaluator()
        
        res = evaluator.evaluate(path="realtime_rawdata/ld_real_time.csv", model=seq2seq)
        df = pd.DataFrame()
        df['PM25'] = res[:, 0]
        df['PM10'] = res[:, 1]
        df['O3'] = res[:, 2]
        df.to_csv("BJ_2425", index=False)

    def LondonForward():
        pass



if __name__ == "__main__":
    bj = bj_upload()

    parser = argparse.ArgumentParser()
    parser.add_argument('-bj_time_interval', action='store', default="2018-04-01-0/2018-04-01-23")
    parser.add_argument('-ld_time_interval', action='store', default="2018-04-01-0/2018-04-01-23")
    parser.add_argument('-bj_model_pt', action='store', default="Model_CheckPoint/seq2seqModel.pt")
    parser.add_argument('-ld_model_pt', action='store', default="beijing/Model_CheckPoint")
    args = parser.parse_args()
    with open ("realtime_rawdata/bj_real_time.csv",'w') as f:
        pass
    bj.get_data(args.bj_time_interval, args.ld_time_interval)
    bj.bjForward(args.bj_model_pt)