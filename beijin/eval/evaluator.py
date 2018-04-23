import torch
import numpy as np
from Dataloader.Seq2Seq_DataLoader import TestDataTransformer

class Evaluator(object):

    def __init__(self):
        pass

    def evaluate(self, path, model):
        data_transformer = TestDataTransformer(path=path, use_cuda=True)
        i = 0
        results = []
        for station in data_transformer.every_station_data:
            s = data_transformer.create_mini_batch(station, use_cuda=True)
            batch = data_transformer.variables_generator(s)
            result = model.forward(batch)
            results.append(result[1].data[0].cpu().numpy())
        results = np.array(results)
        return results.reshape(-1, 3)
