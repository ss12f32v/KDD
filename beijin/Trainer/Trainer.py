import torch

class Trainer(object):

    def __init__(self, model, data_transformer, loggers, learning_rate, use_cuda,
                 checkpoint_name= "Model_CheckPoint/seq2seqModel.pt"
                ):

        self.model = model
        self.train_logger = loggers[0]
        self.valid_logger = loggers[1]
        # record some information about dataset
        self.data_transformer = data_transformer
        self.use_cuda = use_cuda

        # optimizer setting
        self.learning_rate = learning_rate
        self.optimizer= torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()
        self.checkpoint_name = checkpoint_name

    def train(self, num_epochs, batch_size, window_size, pretrained= False, valid_portion= 0.8, time_lag=10):
        
        self.window_size = window_size
        if pretrained:
            self.load_model()

        self.global_step = 0
        for epoch in range(0, num_epochs):
            print("In %d epoch" %(epoch))
            for data in self.data_transformer.every_station_data:

                # validation
                train_data = data[: int(len(data)*valid_portion)]
                valid_data = data[ int(len(data)*valid_portion):]  

                print("Training on %d, Validation with %d" % (len(train_data), len(valid_data)))
                batches, labels = self.data_transformer.create_mini_batch(train_data, batch_size=batch_size, window_size= self.window_size, time_lag=time_lag)
                for input_batch, target_batch in self.data_transformer.variables_generator(batches, labels):

                    self.optimizer.zero_grad()
                    self.model.train()
                    encoder_outputs, decoder_outputs = self.model(input_batch)

                    
                    # calculate the loss and back prop.
                    cur_loss = self.get_loss(encoder_outputs,
                                             decoder_outputs,
                                             input_batch[:,:,0:6].transpose(0, 1),
                                             target_batch.transpose(0, 1))

                    smape_loss = self.smape_loss(encoder_outputs,
                                                 decoder_outputs,
                                                 input_batch[:,:,0:6].transpose(0, 1),
                                                 target_batch.transpose(0, 1)).data[0]
                    cur_loss.backward()                     
                    # logging
                    self.global_step += 1
                    self.tensorboard_log(cur_loss.data[0], smape_loss)
                    if self.global_step % 5 == 0:
                        print("Step: %d, Mse Loss : %4.4f, SMAPE Loss : %f" %(self.global_step, cur_loss.data[0], smape_loss), end='\t')
                    # self.save_model()
                        self.validation(valid_data, batch_size=150, time_lag=time_lag)

                    # optimize
                    self.optimizer.step()

        self.save_model()



    def smape_loss(self, encoder_outputs, decoder_outputs, input_batch, target_batch, val_only=False):  
        if val_only:
            concat_predict = decoder_outputs
            concat_label = target_batch
        concat_predict = torch.cat((encoder_outputs, decoder_outputs), dim= 1)
        concat_label = torch.cat((input_batch, target_batch), dim= 1)
        
        concat_predict = concat_predict[:, :, :3]
        concat_label = concat_label[:, :, :3]
        # print(torch.abs(concat_predict - concat_label).size())
        loss = 2 *  (torch.abs(concat_predict - concat_label).sum(2) /  (torch.abs(concat_predict) + concat_label).sum(2)) # B ＊　Ｔ
        loss = loss.sum()
        loss = loss / (concat_predict.size(0) * concat_predict.size(1))   # Divide the batch size and number of days to get mean 

        return loss

    def get_loss(self, encoder_outputs, decoder_outputs, input_batch, target_batch, val_only=False):  
        if val_only:
            concat_predict = decoder_outputs
            concat_label = target_batch
        concat_predict = torch.cat((encoder_outputs, decoder_outputs), dim= 1)
        concat_label = torch.cat((input_batch, target_batch), dim= 1)
        
        loss = self.criterion(concat_predict, concat_label)
        # for i in range(concat_label.size(1)):
        #     i_timestep_predict = concat_predict[:,i,:].contiguous().view(concat_label.size(0),-1)
        #     i_timestep_label = concat_label[:,i,:].contiguous().view(concat_label.size(0),-1)
        #     loss += self.criterion(i_timestep_predict, i_timestep_label)
        return loss

    def validation(self, valid_data, batch_size, time_lag):
        total_mse_loss = 0
        total_smape_loss = 0
        number_of_batch =0
        self.model.eval()
        batches, labels = self.data_transformer.create_mini_batch(valid_data, batch_size=batch_size, window_size=self.window_size, time_lag=time_lag)
        for input_batch, target_batch in self.data_transformer.variables_generator(batches, labels):
            encoder_outputs, decoder_outputs = self.model(input_batch)

            # calculate the loss and back prop.
            cur_mse_loss = self.get_loss(encoder_outputs,
                                        decoder_outputs,
                                        input_batch[:,:,0:6].transpose(0, 1),
                                        target_batch.transpose(0, 1),
                                        val_only=True).data[0]
            smape_loss = self.smape_loss(encoder_outputs,
                                            decoder_outputs,
                                            input_batch[:,:,0:6].transpose(0, 1),
                                            target_batch.transpose(0, 1),
                                            val_only=True).data[0]
            total_mse_loss += (cur_mse_loss * input_batch.size(1))  # Mulitply Batch number  input_batch size: T * B * H 
            total_smape_loss += (smape_loss * input_batch.size(1))
            number_of_batch += input_batch.size(1)
        total_mse_loss /= number_of_batch
        total_smape_loss /= number_of_batch
        self.tensorboard_log(total_mse_loss, total_smape_loss, valid= True)
        print("Validation, Mse Loss : %4.4f, SMAPE Loss : %f" %(total_mse_loss, total_smape_loss))


    
    def save_model(self):
        torch.save(self.model.state_dict(), self.checkpoint_name)
        print("Model has been saved as %s.\n" % self.checkpoint_name)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.checkpoint_name))
        print("Pretrained model has been loaded.\n")

    def tensorboard_log(self, mse_loss, smape_loss, valid= False):
        info = {
                'mse_loss': mse_loss,
                'smape_loss': smape_loss
            }
        if not valid:
            
            for tag, value in info.items():
                self.train_logger.scalar_summary(tag, value, self.global_step)

        else:
            for tag, value in info.items():
                self.valid_logger.scalar_summary(tag, value, self.global_step)


class RandomTrainer(Trainer):
    
    def __init__(self, *args, **kwargs):
        super(RandomTrainer, self).__init__(*args, **kwargs)

    def train(self, num_epochs, batch_size, window_size, pretrained= False, valid_portion= 0.8, time_lag=10, valid_batch_size=256):
        
        self.window_size = window_size
        if pretrained:
            self.load_model()

        all_station_train_input, all_station_train_label, all_station_valid_input, all_station_valid_label = self.data_transformer.prepare_all_station_data(
            all_station_data=self.data_transformer.every_station_data,
            training_portion=valid_portion,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            window_size=self.window_size,
            time_lag=time_lag,
            shuffle_input=True)

        print("Training on %d, Validation with %d" % (len(all_station_train_input) * batch_size, len(all_station_valid_input) * valid_batch_size))
        self.global_step = 0
        for epoch in range(1, num_epochs + 1):
            print("In %d epoch" %(epoch))
            for input_batch, target_batch in self.data_transformer.variables_generator(all_station_train_input, all_station_train_label):
                self.optimizer.zero_grad()
                self.model.train()
                encoder_outputs, decoder_outputs = self.model(input_batch)

                # calculate the loss and back prop.
                cur_loss = self.get_loss(encoder_outputs,
                                            decoder_outputs,
                                            input_batch[:,:,0:6].transpose(0, 1),
                                            target_batch.transpose(0, 1))

                smape_loss = self.smape_loss(encoder_outputs,
                                                decoder_outputs,
                                                input_batch[:,:,0:6].transpose(0, 1),
                                                target_batch.transpose(0, 1)).data[0]
                cur_loss.backward()                     
                # logging
                self.global_step += 1
                self.tensorboard_log(cur_loss.data[0], smape_loss)
                if self.global_step % 5 == 0:
                    print("[Epoch #%d] Step: %d, Mse Loss : %4.4f, SMAPE Loss : %f" %(epoch, self.global_step, cur_loss.data[0], smape_loss), end='\t')
                # self.save_model()
                    self.validation(all_station_valid_input, all_station_valid_label, valid_batch_size, time_lag)

                # optimize
                self.optimizer.step()

            self.save_model()

    def validation(self, valid_inputs, valid_labels, batch_size, time_lag):
        total_mse_loss = 0
        total_smape_loss = 0
        number_of_batch =0
        self.model.eval()
        for input_batch, target_batch in self.data_transformer.variables_generator(valid_inputs, valid_labels):
            encoder_outputs, decoder_outputs = self.model(input_batch)

            # calculate the loss and back prop.
            cur_mse_loss = self.get_loss(encoder_outputs,
                                        decoder_outputs,
                                        input_batch[:,:,0:6].transpose(0, 1),
                                        target_batch.transpose(0, 1),
                                        val_only=True).data[0]
            smape_loss = self.smape_loss(encoder_outputs,
                                            decoder_outputs,
                                            input_batch[:,:,0:6].transpose(0, 1),
                                            target_batch.transpose(0, 1),
                                            val_only=True).data[0]
            total_mse_loss += (cur_mse_loss * input_batch.size(1))  # Mulitply Batch number  input_batch size: T * B * H 
            total_smape_loss += (smape_loss * input_batch.size(1))
            number_of_batch += input_batch.size(1)
        total_mse_loss /= number_of_batch
        total_smape_loss /= number_of_batch
        self.tensorboard_log(total_mse_loss, total_smape_loss, valid= True)
        print("Validation, Mse Loss : %4.4f, SMAPE Loss : %f" %(total_mse_loss, total_smape_loss))