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

        self.__train_events = train_data['events']
        self.__train_event_times = train_data['times']
        # self.__test_data = test_data

        self.__num_of_nodes = self.__train_events.shape[0]
        self.__num_of_bills = self.__train_events.shape[1]

        self.__dim = dim
        self.__number_of_variables = order + 1

        self.__epochs_num = epochs_num
        self.__lr = learning_rate

        self.samples_num = samples_num

        # Set the models parameters
        self.__beta = torch.nn.Parameter(torch.rand(size=(self.__num_of_bills,)), requires_grad=False)  # bias for bills
        self.__gamma = torch.nn.Parameter(torch.rand(size=(self.__num_of_nodes,)), requires_grad=False)  # bias for politicians
        self.__z_b = torch.nn.Parameter(torch.rand(size=(self.__num_of_bills, self.__dim), ), requires_grad=True)
        self.__z_p = [torch.nn.Parameter(torch.rand(size=(self.__num_of_nodes, self.__dim)), requires_grad=False) for _ in
             range(self.__number_of_variables)]
        self.__z_p = torch.nn.ParameterList(self.__z_p)

        # Store the factorial terms
        self.__factorials = [float(math.factorial(o)) for o in range(self.__number_of_variables)]

        self.__pdist = torch.nn.PairwiseDistance(p=2, keepdim=False)

    def get_latent_variable_at(self, i, order_index, t):

        z = torch.zeros(size=(self.__dim,), dtype=torch.float)
        for o in range(self.__number_of_variables - order_index):
            z += self.__z_p[order_index + o][i, :] * (t ** o) / math.factorial(o)

        return z

    def get_all_politician_positions(self, t):

        z = torch.zeros(size=(self.__num_of_nodes, self.__dim,), dtype=torch.float)
        for o in range(self.__number_of_variables):
            z += self.__z_p[o] * (t ** o) / math.factorial(o)

        return z

    def get_position_of_politician_at(self, i, t):

        return self.get_latent_variable_at(i=i, order_index=0, t=t)

    def get_position_of_bill(self, bill_idx):

        return self.__z_b[bill_idx, :]

    def get_distance(self, politician_idx, bill_idx, t):

        z_p = self.get_position_of_politician_at(i=politician_idx, t=t)
        z_b = self.get_position_of_bill(bill_idx=bill_idx)

        temp = z_p - z_b
        return torch.sqrt(torch.dot(temp, temp))

    def get_bill_bias(self, bill_idx):

        return self.__beta[bill_idx]

    def get_politican_bias(self, politician_idx):

        return self.__gamma[politician_idx]

    def compute_likelihood_for(self, politician_idx, bill_idx, t):

        return sigmoid( self.get_politican_bias(politician_idx) + self.get_bill_bias(bill_idx) - self.get_distance(politician_idx, bill_idx, t) )

    def compute_log_likelihood(self, bill_idx, bill_events, bill_time):

        log_likelihood = 0.
        for node in range(self.__num_of_nodes):

            p = self.compute_likelihood_for(politician_idx=node, bill_idx=bill_idx, t=bill_time)
            if bill_events[node]:
                log_likelihood += torch.log( p )
            else:
                log_likelihood += torch.log( 1. - p)

        return log_likelihood

    def forward(self, events, events_time):

        neg_logLikelihood = 0.
        for bill_idx in range(self.__num_of_bills):

            s = self.compute_log_likelihood(bill_idx=bill_idx, bill_events=events[:, bill_idx],
                                                             bill_time=events_time[bill_idx])
            # print(bill_idx, s)
            neg_logLikelihood += s

        return -neg_logLikelihood

    def set_gradients_of_latent_variables(self, index, value=True):

        self.__z_p[index].requires_grad = value

    def set_gradient_of_bias_terms(self, value=True):

        self.__beta.requires_grad = value
        self.__gamma.requires_grad = value

    def learn(self, type="seq"):

        # List for storing the training and testing set losses
        train_loss_list, test_loss_list = [], []

        # Learns the parameters sequentially
        if type == "seq":

            for inx in [i for i in range(1, self.__number_of_variables + 1)] + [0]:

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
                    train_loss_list.append(train_loss.item() / self.__num_of_bills)

                    # self.eval()
                    # with torch.no_grad():
                    #     test_loss = self(data=self.__testSet)
                    #     testLoss.append(test_loss / len(self.__testSet))

                    if epoch % 50 == 0:
                        print(f"Epoch {epoch + 1} train loss: {train_loss.item() / self.__num_of_bills}")
                        # print(f"Epoch {epoch + 1} test loss: {test_loss / len(self.__testSet)}")

        return train_loss_list, test_loss_list

