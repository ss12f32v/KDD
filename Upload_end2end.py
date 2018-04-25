

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
    def get_data(self, bj_time_interval, bj_csv):
        bj_url = 'https://biendata.com/competition/airquality/bj/' + bj_time_interval +'/2k0d1d8'
        print(bj_url)
        respones= requests.get(bj_url)
        with open (bj_csv,'w') as f:
            f.write(respones.text)
            f.close()
        data = pd.read_csv(bj_csv).drop(['NO2_Concentration', 'CO_Concentration', 'SO2_Concentration'], axis=1)
        data.to_csv(bj_csv, index= False)
        
    
    def bjForward(self, bj_model_pt, bj_csv):   
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
        
        res = evaluator.evaluate(path= bj_csv, model= seq2seq)
        print(res)
        df = pd.DataFrame()
        df['PM25'] = res[:, 0]
        df['PM10'] = res[:, 1]
        df['O3'] = res[:, 2]
        # df.to_csv("BJ_2425", index=False)
        return df

class ld_upload():

    def __init__(self):
        pass
    def get_data(self, ld_time_interval, ld_csv):
        ld_url = 'https://biendata.com/competition/airquality/ld/' + ld_time_interval +'/2k0d1d8'
        print(ld_url)
        respones= requests.get(ld_url)
        with open (ld_csv,'w') as f:
            f.write(respones.text)
            f.close()
        data = pd.read_csv(ld_csv).drop(['NO2_Concentration', 'CO_Concentration', 'SO2_Concentration'], axis=1)
        data.to_csv(ld_csv, index= False)
        
        

    def ldForward(self, ld_model_pt, ld_csv):   
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

        seq2seq.load_state_dict(torch.load(ld_model_pt))
        
        evaluator = Evaluator()
        
        res = evaluator.evaluate(path= ld_csv, model= seq2seq, london= True)
        print(res)
        df = pd.DataFrame()
        df['PM25'] = res[:, 0]
        df['PM10'] = res[:, 1]
        # df['O3'] = res[:, 2]
        # df.to_csv("BJ_2425", index=False)
        return df


if __name__ == "__main__":
    bj = bj_upload()
    ld = ld_upload()
    parser = argparse.ArgumentParser()
    parser.add_argument('-bj_time_interval', action='store', default="2018-04-01-0/2018-04-24-13")
    parser.add_argument('-bj_model_pt', action='store', default="beijing/Model_CheckPoint/seq2seqModel.pt")
    parser.add_argument('-bj_csv', action='store', default="realtime_rawdata/bj_real_time.csv")


    parser.add_argument('-ld_time_interval', action='store', default="2018-04-01-0/2018-04-24-12") 
    parser.add_argument('-ld_model_pt', action='store', default="london/Model_CheckPoint/seq2seqModel.pt")
    parser.add_argument('-ld_csv', action='store', default="realtime_rawdata/ld_real_time.csv")

    
    args = parser.parse_args()
    with open (args.bj_csv,'w') as f:
        pass
    with open (args.ld_csv,'w') as f:
        pass

    bj.get_data(args.bj_time_interval, args.bj_csv)
    bj_df = bj.bjForward(args.bj_model_pt, args.bj_csv)

    ld.get_data(args.ld_time_interval, args.ld_csv)
    ld_df = ld.ldForward(args.ld_model_pt, args.ld_csv)

   # Make upload submission
    concat = pd.concat([bj_df,ld_df]).fillna(0)


    sub = pd.read_csv("beijing/Data/sample_submission.csv")
    sub['PM2.5'] = concat.loc[:,'PM25'].as_matrix()
    sub['PM10'] = concat.loc[:,'PM10'].as_matrix()
    sub['O3'] = concat.loc[:,'O3'].as_matrix()
    sub.to_csv("realtime_rawdata/Submission.csv",index= False)

    # Uploading 
    # files={'files': open("realtime_rawdata/Submission.csv",'rb')}

    # data = {
    #     "user_id": "ss12f32v",   #user_id is your username which can be found on the top-right corner on our website when you logged in.
    #     "team_token": "e53b7c76ef0c3a9b758c24788188b6fa7ce49e211fa0d6bfd18327bb77a770b1", #your team_token.
    #     "description": 'your description',  #no more than 40 chars.
    #     "filename": "fix decoder", #your filename
    # }

    # url = 'https://biendata.com/competition/kdd_2018_submit/'

    # response = requests.post(url, files=files, data=data)

    # print(response.text)


