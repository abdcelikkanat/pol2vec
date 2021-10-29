import torch
import math
import time
import numpy as np
from torch import sigmoid
from torch.nn.functional import logsigmoid as log_sigmoid
from torch.distributions.normal import Normal


class Pol2VecMulti(torch.nn.Module):

    def __init__(self, train_data, dim, order, metric, metric_param=None, epochs_num=100,
                 learning_rate=0.1, sigma_grad=False, seed=123, verbose=True):

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

        self.__metric = metric
        self.__metric_param = metric_param

        # Optimization parameters
        self.__epochs_num = epochs_num
        self.__lr = learning_rate

        # Set the models parameters
        self.__gamma_cols = torch.nn.Parameter(2*torch.rand(size=(self.__col_size,))-1., requires_grad=False)
        self.__gamma_rows = torch.nn.Parameter(2*torch.rand(size=(self.__row_size,))-1., requires_grad=False)
        self.__z_cols = torch.nn.Parameter(2*torch.rand(size=(self.__col_size, self.__dim))-1., requires_grad=True)
        # self.__z_rows = [torch.rand(size=(self.__row_size, self.__dim)) for _ in range(self.__var_size)]
        # self.__z_rows = torch.nn.ParameterList([torch.nn.Parameter(var, requires_grad=False) for var in self.__z_rows])
        self.__z_rows = torch.nn.Parameter(2*torch.rand(size=(self.__var_size, self.__row_size, self.__dim))-1., requires_grad=True)

        # Class num is the number of different labels except 0
        self.__class_num = len(set(e for event_mat in self.__train_events for e in event_mat.data.tolist()))

        self.__b = torch.nn.Parameter(torch.tensor(np.linspace(-1., 1., self.__class_num-1).tolist(), dtype=torch.float))
        self.__sigma = torch.nn.Parameter(torch.tensor([1.]), requires_grad=sigma_grad)

        # A matrix for Mahalanobis distance
        self.__L = 2 * torch.zeros(size=(self.__col_size, self.__dim, self.__dim)) - 1.
        for col_idx in range(self.__col_size):
            if self.__metric_param == "diag" or metric_param == "diagonal":
                self.__L[col_idx, :, :] = torch.diag(2 * torch.rand(size=(self.__dim, )) - 1.)
            elif self.__metric_param == "full":
                self.__L[col_idx, :, :] = torch.tril(2 * torch.rand(size=(self.__dim, self.__dim)) - 1.)
            else:
                raise ValueError("Invalid metric parameter value!")
        self.__L = torch.nn.Parameter(self.__L, requires_grad=True)

        print(self.__L.shape)

        # Store the factorial terms
        self.__factorials = [float(math.factorial(o)) for o in range(self.__var_size)]

        # Define the normal distribution
        self.__normal = Normal(0, 1)

        # Define a big number
        self.__big_number = 1e+5

        #
        self.__b.data[(self.__class_num-1)//2] = 0.

        if verbose:
            print("Number of classes: {}".format(self.__class_num))

        # Distance function
        self.__pdist = torch.nn.PairwiseDistance(p=2)

    def get_variable_at(self, i, order_index, t):

        z = torch.zeros(size=(self.__dim,), dtype=torch.float)
        for o in range(self.__var_size - order_index):
            z += self.__z_rows[order_index + o][i, :] * (t ** o) / math.factorial(o)

        return z

    def get_position_of_row_at(self, i, t):

        return self.get_variable_at(i=i, order_index=0, t=t)

    def get_all_row_positions(self, mat_indices, col_times):

        # z = torch.zeros(size=(len(mat_indices), self.__dim), dtype=torch.float)
        t0 = time.time()
        col_times_mat = torch.tensor([(np.asarray(col_times) ** o) / self.__factorials[o] for o in range(self.__var_size)], dtype=torch.float )
        # print("Col times: {}".format(time.time() - t0))

        t0 = time.time()
        # for idx, mat_idx in enumerate(mat_indices):
        #     z[idx, :] = torch.matmul(col_times_mat[:, mat_idx[1]], self.__z_rows[:, mat_idx[0], :])

        z = torch.matmul(self.__z_rows.transpose(0, 2), col_times_mat).permute(1, 2, 0) #.reshape(-1, self.__dim)
        z = z[mat_indices.T[0], mat_indices.T[1], :]
        # print("z assignment: {}".format(time.time() - t0))
        # print(z.shape)
        return z

    def get_position_of_column(self, col_idx):

        return self.__z_cols[col_idx, :]

    def get_distance(self, row_idx, col_idx, t):

        z_row = self.get_position_of_row_at(i=row_idx, t=t)
        z_col = self.get_position_of_column(col_idx=col_idx)

        temp = z_row - z_col
        return torch.sqrt(torch.dot(temp, temp))

    def get_all_pairwise_distances(self, mat_indices, col_idx_list, col_times):

        t0 = time.time()
        z_rows = self.get_all_row_positions(mat_indices=mat_indices, col_times=col_times)

        if self.__metric == "euc" or self.__metric == "euclidean":

            return self.__pdist(z_rows, self.__z_cols[mat_indices.T[1], :])

        elif self.__metric == "mah" or self.__metric == "mahalanobis":

            chosen_L = self.__L[col_idx_list, :, :]
            # precision_matrix = torch.matmul(chosen_L, chosen_L.transpose(0, 1))

            delta = z_rows - self.__z_cols[mat_indices.T[1], :]
            T = torch.matmul(delta.unsqueeze(1), chosen_L[mat_indices.T[1], :, :])

            return torch.norm(T.squeeze(1), p=2, dim=1)

        else:

            raise ValueError("Invalid metric name!")

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

    def compute_all_f(self, mat_indices, col_idx_list, col_times):

        dist = -self.get_all_pairwise_distances(mat_indices=mat_indices, col_idx_list=col_idx_list, col_times=col_times)

        dist += self.get_all_row_bias(indices=mat_indices.T[0])
        dist += self.get_all_col_bias(indices=mat_indices.T[1])

        return dist

    def compute_all_log_likelihood(self, event_mat, col_idx_list, col_times):
        '''

        :param event_mat: row_size x col_size matrix
        :param col_times:
        :return:
        '''

        t0 = time.time()
        mat_indices = event_mat.nonzero()
        # print("Step0: ", time.time() - t0)

        t0 = time.time()
        temp = self.compute_all_f(mat_indices=mat_indices, col_idx_list=col_idx_list, col_times=col_times)
        # print("Step1: ", time.time() - t0)
        # print(temp)
        t0 = time.time()
        theta = torch.hstack((torch.tensor([-self.__big_number]), self.__b, torch.tensor([self.__big_number])))
        # print("Step2: ", time.time() - t0)

        t0 = time.time()
        y0, y1 = [], []
        for row_idx, col_idx in mat_indices:
            y0.append(event_mat[row_idx, col_idx] - 1)
            y1.append(event_mat[row_idx, col_idx])
        # y0 = [event_mat[row_idx, col_idx] - 1 for row_idx, col_idx in mat_indices]
        # y1 = [event_mat[row_idx, col_idx] for row_idx, col_idx in mat_indices]
        # print("Step3: ", time.time() - t0)

        t0 = time.time()
        ll = torch.log(self.__normal.cdf((theta[y1] - temp)/self.__sigma) - self.__normal.cdf((theta[y0] - temp)/self.__sigma))
        # print("Step4: ", time.time() - t0)

        return ll.sum()

    def forward(self, events, col_idx_list, events_time):

        return -self.compute_all_log_likelihood(event_mat=events, col_idx_list=col_idx_list, col_times=events_time)

    def set_gradients_of_latent_variables(self, index, value=True):

        # self.__z_rows[index].requires_grad = value
        pass

    def __set_gradients(self, inx):

        #
        self.__b.grad[(self.__class_num-1)//2] = 0

        #
        for col_idx in range(self.__col_size):
            if self.__metric_param == "full":

                self.__L.grad[col_idx, torch.triu_indices(row=self.__dim, col=self.__dim, offset=1)] = 0

            if self.__metric_param == "diag" or self.__metric_param == "diagonal":

                self.__L.grad[col_idx, range(self.__dim),  range(self.__dim)] = 0



        #
        if inx > 0:
            self.__z_rows.grad[inx:, :, :] = 0

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
                    # print("forward")
                    total_train_loss = 0.
                    counter = 0
                    for batch_train_events, col_indices, batch_event_times in zip(self.__train_events, self.__train_col_indices, self.__train_event_times):
                        # print(counter)
                        counter += 1
                        train_loss = self.forward(events=torch.tensor(batch_train_events.todense(), dtype=torch.int8), col_idx_list=col_indices, events_time=batch_event_times)
                        total_train_loss += train_loss
                    # Backward pass
                    total_train_loss.backward()

                    self.__set_gradients(inx=inx)

                    # Step
                    optimizer.step()
                    # print(self.__b)
                    # print("post-forward")
                    # Append the loss
                    train_loss_list.append(total_train_loss.item() / self.__col_size)

                    # self.eval()
                    # with torch.no_grad():
                    #     test_loss = self(data=self.__testSet)
                    #     testLoss.append(test_loss / len(self.__testSet))

                    if epoch % 50 == 0:
                        print(f"Epoch {epoch + 1} train loss: {total_train_loss.item() / self.__col_size}")
                        # print(f"Epoch {epoch + 1} test loss: {test_loss / len(self.__testSet)}")

        return train_loss_list, test_loss_list

    def get_label_assignments(self, mat_indices, col_times):

        num_of_elements = len(mat_indices)
        values = self.compute_all_f(mat_indices, col_times)

        assigned_y = [0 for _ in range(num_of_elements)]
        for i in range(num_of_elements):
            c = 0
            for b in self.__b:
                if values[i] <= b:
                    break
                c += 1

            assigned_y[i] = c

        return assigned_y
