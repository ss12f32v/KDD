import torch

class Trainer(object):

    def __init__(self, model, data_transformer, learning_rate, use_cuda,
                 checkpoint_name= "Model_CheckPoint/seq2seqModel.pt"
                ):

        self.model = model

        # record some information about dataset
        self.data_transformer = data_transformer
        self.use_cuda = use_cuda

        # optimizer setting
        self.learning_rate = learning_rate
        self.optimizer= torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()
        self.checkpoint_name = checkpoint_name

    def train(self, num_epochs, batch_size, window_size, pretrained=False):

        if pretrained:
            self.load_model()

        step = 0

        for epoch in range(0, num_epochs):
            for data in self.data_transformer.every_station_data:
                for input_batch, target_batch in self.data_transformer.mini_batch_generator(data, batch_size=batch_size, window_size= window_size):

                    self.optimizer.zero_grad()
                    encoder_outputs, decoder_outputs = self.model(input_batch)

                    # calculate the loss and back prop.
                    cur_loss = self.get_loss(encoder_outputs,
                                             decoder_outputs,
                                             input_batch.transpose(0, 1),
                                             target_batch.transpose(0, 1))

                    smape_loss = self.smape_loss(encoder_outputs,
                                                 decoder_outputs,
                                                 input_batch.transpose(0, 1),
                                                 target_batch.transpose(0, 1))
                                             
                    # logging
                    step += 1
                    if step % 1 == 0:
                        print("Step: %d, Mse Loss : %f, SMAPE Loss : %f" %(step,cur_loss.data[0],smape_loss.data[0] ))
                        # self.save_model()
                    cur_loss.backward()

                    # optimize
                    self.optimizer.step()

        self.save_model()

    def smape_loss(self, encoder_outputs, decoder_outputs, input_batch, target_batch):  

        concat_predict = torch.cat((encoder_outputs, decoder_outputs), dim= 1)
        concat_label = torch.cat((input_batch, target_batch), dim= 1)
        print(concat_predict.size())
        loss = 2 * torch.abs(concat_predict - concat_label).sum() /  (concat_predict + concat_label).sum()
        loss = loss / (concat_predict.size(0) * concat_predict.size(1))   # Divide the batch size and number of days to get mean 

        return loss

    def get_loss(self, encoder_outputs, decoder_outputs, input_batch, target_batch):  
        
        concat_predict = torch.cat((encoder_outputs, decoder_outputs), dim= 1)
        concat_label = torch.cat((input_batch, target_batch), dim= 1)

        
        loss = self.criterion(concat_predict,concat_label)
        # for i in range(concat_label.size(1)):
        #     i_timestep_predict = concat_predict[:,i,:].contiguous().view(concat_label.size(0),-1)
        #     i_timestep_label = concat_label[:,i,:].contiguous().view(concat_label.size(0),-1)
        #     loss += self.criterion(i_timestep_predict, i_timestep_label)
        return loss
    def save_model(self):
        torch.save(self.model.state_dict(), self.checkpoint_name)
        print("Model has been saved as %s.\n" % self.checkpoint_name)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.checkpoint_name))
        print("Pretrained model has been loaded.\n")

    def tensorboard_log(self):
        pass

    def evaluate(self, words):
        # make sure that words is list
        if type(words) is not list:
            words = [words]

        # transform word to index-sequence
        eval_var = self.data_transformer.evaluation_batch(words=words)
        decoded_indices = self.model.evaluation(eval_var)
        results = []
        for indices in decoded_indices:
            results.append(self.data_transformer.vocab.indices_to_sequence(indices))
        return results
