import torch
import math
import time
import numpy as np
from torch import sigmoid
from torch.nn.functional import logsigmoid as log_sigmoid


class Pol2VecBinary(torch.nn.Module):

    def __init__(self, train_data, dim, order, epochs_num, learning_rate, samples_num=10, seed=123):
        super().__init__()

        # Set the seed value
        self.__seed = seed
        torch.manual_seed(self.__seed)
        np.random.seed(self.__seed)

        self.__train_events = train_data['events']
        self.__train_col_indices = train_data['col_indices']
        self.__train_event_times = train_data['times']

        # print(self.__train_event_times)

        # self.__test_data = test_data

        self.__row_size = self.__train_events[0].shape[0] #None #self.__train_events.shape[0]
        self.__col_size = sum([len(event_times) for event_times in self.__train_event_times])  #None #self.__train_events.shape[1]

        self.__dim = dim
        self.__var_size = order + 1

        # Optimization parameters
        self.__epochs_num = epochs_num
        self.__lr = learning_rate

        # Set the models parameters
        self.__gamma_cols = torch.nn.Parameter(torch.rand(size=(self.__col_size,)), requires_grad=False)
        self.__gamma_rows = torch.nn.Parameter(torch.rand(size=(self.__row_size,)), requires_grad=False)
        self.__z_cols = torch.nn.Parameter(torch.rand(size=(self.__col_size, self.__dim), ), requires_grad=True)
        self.__z_rows = [torch.rand(size=(self.__row_size, self.__dim)) for _ in range(self.__var_size)]
        self.__z_rows = torch.nn.ParameterList([torch.nn.Parameter(var, requires_grad=False) for var in self.__z_rows])

        # Store the factorial terms
        self.__factorials = [float(math.factorial(o)) for o in range(self.__var_size)]

    def get_variable_at(self, i, order_index, t):

        z = torch.zeros(size=(self.__dim,), dtype=torch.float)
        for o in range(self.__var_size - order_index):
            z += self.__z_rows[order_index + o][i, :] * (t ** o) / math.factorial(o)

        return z

    def get_position_of_row_at(self, i, t):

        return self.get_variable_at(i=i, order_index=0, t=t)

    def get_all_row_positions(self, col_times, col_indices):

        z = torch.zeros(size=(self.__row_size, self.__dim, len(col_times) ), dtype=torch.float)
        col_times_mat = torch.tensor([(np.asarray(col_times) ** o) / math.factorial(o) for o in range(self.__var_size)], dtype=torch.float )

        for idx, col_t in enumerate(col_times):
            for o in range(self.__var_size):
                z[:, :, idx] += self.__z_rows[o] * col_times_mat[o, idx]

        return z

    def get_position_of_column(self, col_idx):

        return self.__z_cols[col_idx, :]

    def get_distance(self, row_idx, col_idx, t):

        z_row = self.get_position_of_row_at(i=row_idx, t=t)
        z_col = self.get_position_of_column(col_idx=col_idx)

        temp = z_row - z_col
        return torch.sqrt(torch.dot(temp, temp))

    def get_all_pairwise_distances(self, col_indices, col_times):

        z_rows = self.get_all_row_positions(col_times=col_times, col_indices=col_indices)  # row_size x dim x col_size
        # print(z_rows.permute(2, 0, 1).shape)
        # print(self.__z_cols[col_indices, :].unsqueeze(1).shape)
        # cdsit output matrix = col_size x row_size x 1
        return torch.cdist(z_rows.permute(2, 0, 1), self.__z_cols[col_indices, :].unsqueeze(1), p=2).squeeze().transpose(1, 0)

    def get_col_bias(self, col_idx):

        return self.__gamma_cols[col_idx]

    def get_row_bias(self, row_idx):

        return self.__gamma_rows[row_idx]

    def get_all_col_bias(self, indices=None):

        if indices is None:
            return self.__gamma_cols
        else:
            return self.__gamma_cols[indices]

    def get_all_row_bias(self, indices=None):

        if indices is None:
            return self.__gamma_rows
        else:
            return self.__gamma_rows[indices]

    def compute_likelihood_for(self, row_idx, col_idx, t):

        return sigmoid( self.get_row_bias(row_idx) + self.get_col_bias(col_idx) - self.get_distance(row_idx, col_idx, t) )

    def compute_all_likelihood(self, col_indices, col_times):

        dist = -self.get_all_pairwise_distances(col_indices=col_indices, col_times=col_times)

        dist += self.get_all_row_bias()[:, None]
        dist += self.get_all_col_bias(indices=col_indices)[None, :]

        return sigmoid(dist)

    def compute_log_likelihood(self, col_idx, col_events, col_time):

        log_likelihood = 0.
        for node in range(self.__row_size):

            p = self.compute_likelihood_for(row_idx=node, col_idx=col_idx, t=col_time)
            if col_events[node]:
                log_likelihood += torch.log( p )
            else:
                log_likelihood += torch.log( 1. - p)

        return log_likelihood

    def compute_all_log_likelihood(self, event_mat, col_indices, col_times):
        '''

        :param event_mat: row_size x col_size matrix
        :param col_times:
        :return:
        '''

        temp = self.compute_all_likelihood(col_indices=col_indices, col_times=col_times)
        # print(temp.shape)
        # print(event_mat.shape)
        p_mat = (1-event_mat) - (1 - 2*event_mat) * temp

        return torch.sum(torch.log(p_mat))

    def forward(self, events, col_indices, events_time):

        return -self.compute_all_log_likelihood(event_mat=events, col_indices=col_indices, col_times=events_time)

    def set_gradients_of_latent_variables(self, index, value=True):

        self.__z_rows[index].requires_grad = value

    def set_gradient_of_bias_terms(self, value=True):

        self.__gamma_rows.requires_grad = value
        self.__gamma_cols.requires_grad = value

    def learn(self, type="seq"):

        # List for storing the training and testing set losses
        train_loss_list, test_loss_list = [], []

        # Learns the parameters sequentially
        if type == "seq":

            for inx in [i for i in range(1, self.__var_size + 1)] + [0]:

                print("Index: {}".format(inx))

                if inx > 0:
                    self.set_gradients_of_latent_variables(inx - 1)
                else:
                    self.set_gradient_of_bias_terms()

                # Define the optimizer
                optimizer = torch.optim.Adam(self.parameters(), lr=self.__lr)

                for epoch in range(self.__epochs_num):
                    # print("Epoch: {}".format(epoch))
                    self.train()

                    for param in self.parameters():
                        param.grad = None

                    # Forward pass
                    print("forward")
                    total_train_loss = 0.
                    for batch_train_events, col_indices, batch_event_times in zip(self.__train_events, self.__train_col_indices, self.__train_event_times):
                        train_loss = self.forward(events=torch.tensor(batch_train_events.todense(), dtype=torch.int8),
                                                  col_indices=col_indices, events_time=batch_event_times)
                        total_train_loss += train_loss.item()
                        # Backward pass
                        train_loss.backward()
                        # Step
                        optimizer.step()

                    print("post-forward")
                    # Append the loss
                    train_loss_list.append(total_train_loss / self.__col_size)

                    # self.eval()
                    # with torch.no_grad():
                    #     test_loss = self(data=self.__testSet)
                    #     testLoss.append(test_loss / len(self.__testSet))

                    if epoch % 50 == 0:
                        print(f"Epoch {epoch + 1} train loss: {total_train_loss.item() / self.__col_size}")
                        # print(f"Epoch {epoch + 1} test loss: {test_loss / len(self.__testSet)}")

        return train_loss_list, test_loss_list

