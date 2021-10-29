import torch
import math
import time
import numpy as np
from torch import sigmoid
from torch.nn.functional import logsigmoid as log_sigmoid


class Pol2Vec(torch.nn.Module):

    def __init__(self, train_data, dim, order, epochs_num, learning_rate, samples_num=10, seed=123):
        super().__init__()

        # Set the seed value
        self.__seed = seed
        torch.manual_seed(self.__seed)
        np.random.seed(self.__seed)

        self.__train_events = torch.tensor(train_data['events'], dtype=torch.int8)
        self.__train_event_times = train_data['times']

        # print(self.__train_event_times)

        # self.__test_data = test_data

        self.__row_size = self.__train_events.shape[0]
        self.__col_size = self.__train_events.shape[1]

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

        self.__pdist = torch.nn.PairwiseDistance(p=2, keepdim=False)

    def get_variable_at(self, i, order_index, t):

        z = torch.zeros(size=(self.__dim,), dtype=torch.float)
        for o in range(self.__var_size - order_index):
            z += self.__z_rows[order_index + o][i, :] * (t ** o) / math.factorial(o)

        return z

    def get_position_of_row_at(self, i, t):

        return self.get_variable_at(i=i, order_index=0, t=t)

    def get_all_row_positions(self, col_times):

        z = torch.zeros(size=(self.__row_size, self.__dim, self.__col_size ), dtype=torch.float)
        col_times_mat = torch.tensor([(np.asarray(col_times) ** o) / math.factorial(o) for o in range(self.__var_size)], dtype=torch.float )

        for col_idx, col_t in enumerate(col_times):
            for o in range(self.__var_size):
                z[:, :, col_idx] += self.__z_rows[o] * col_times_mat[o, col_idx]

        return z

    def get_position_of_column(self, col_idx):

        return self.__z_cols[col_idx, :]

    def get_distance(self, row_idx, col_idx, t):

        z_row = self.get_position_of_row_at(i=row_idx, t=t)
        z_col = self.get_position_of_column(col_idx=col_idx)

        temp = z_row - z_col
        return torch.sqrt(torch.dot(temp, temp))

    def get_all_pairwise_distances(self, col_times):

        z_rows = self.get_all_row_positions(col_times=col_times)  # row_size x dim x col_size

        # cdsit output matrix = col_size x row_size x 1
        return torch.cdist(z_rows.permute(2, 0, 1), self.__z_cols.unsqueeze(1), p=2).squeeze().transpose(1, 0)

    def get_col_bias(self, col_idx):

        return self.__gamma_cols[col_idx]

    def get_row_bias(self, row_idx):

        return self.__gamma_rows[row_idx]

    def get_all_col_bias(self):

        return self.__gamma_cols

    def get_all_row_bias(self):

        return self.__gamma_rows

    def compute_likelihood_for(self, row_idx, col_idx, t):

        return sigmoid( self.get_row_bias(row_idx) + self.get_col_bias(col_idx) - self.get_distance(row_idx, col_idx, t) )

    def compute_all_likelihood(self, col_times):

        dist = -self.get_all_pairwise_distances(col_times=col_times)

        dist += self.get_all_row_bias()[:, None]
        dist += self.get_all_col_bias()[None, :]

        return sigmoid(dist)

    def compute_log_likelihood(self, col_idx, col_events, col_time):
        self.compute_all_likelihood(col_events)
        log_likelihood = 0.
        for node in range(self.__row_size):

            p = self.compute_likelihood_for(row_idx=node, col_idx=col_idx, t=col_time)
            if col_events[node]:
                log_likelihood += torch.log( p )
            else:
                log_likelihood += torch.log( 1. - p)

        return log_likelihood

    def compute_all_log_likelihood(self, event_mat, col_times):
        '''

        :param event_mat: row_size x col_size matrix
        :param col_times:
        :return:
        '''

        p_mat = (1-event_mat) - (1 - 2*event_mat) * self.compute_all_likelihood(col_times=col_times)

        return torch.sum(torch.log(p_mat))

    def forward(self, events, events_time):

        neg_logLikelihood = 0.
        # for col_idx in range(self.__col_size):
        #
        #     s = self.compute_log_likelihood(col_idx=col_idx, col_events=events[:, col_idx],
        #                                                      col_time=events_time[col_idx])
        #     # print(bill_idx, s)
        #     neg_logLikelihood += s

        neg_logLikelihood = self.compute_all_log_likelihood(event_mat=self.__train_events, col_times=self.__train_event_times)

        return -neg_logLikelihood

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

                    self.train()

                    for param in self.parameters():
                        param.grad = None

                    # Forward pass
                    train_loss = self.forward(events=self.__train_events, events_time=self.__train_event_times)

                    # print(train_loss)
                    # Backward pass
                    train_loss.backward()
                    # Step
                    optimizer.step()
                    # Append the loss
                    train_loss_list.append(train_loss.item() / self.__col_size)

                    # self.eval()
                    # with torch.no_grad():
                    #     test_loss = self(data=self.__testSet)
                    #     testLoss.append(test_loss / len(self.__testSet))

                    if epoch % 50 == 0:
                        print(f"Epoch {epoch + 1} train loss: {train_loss.item() / self.__col_size}")
                        # print(f"Epoch {epoch + 1} test loss: {test_loss / len(self.__testSet)}")

        return train_loss_list, test_loss_list

