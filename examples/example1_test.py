import os
import matplotlib.pyplot as plt
from models.pol2vecbinary import *
from datasets.loader import *
from utilities.animation import *
from utilities.complete_motion import CompleteMotion
torch.set_num_threads(16)


def batching(event_mat, event_times, batch_size):
    event_mat_list = []
    event_times_list = []
    col_indices_list = []

    for i in range(0, event_mat.shape[1], batch_size):
        event_mat_list.append(event_mat[:, i:i + batch_size])
        col_indices_list.append(list(range(i, i+batch_size)))
        event_times_list.append(event_times[i:i + batch_size])

    return event_mat_list, col_indices_list, event_times_list

# Define the global variables
script_folder = os.path.dirname(__file__)
base_folder = os.path.realpath(os.path.join(script_folder, ".."))
dataset_folder = os.path.realpath(os.path.join(base_folder, "datasets"))

# Set the dataset name
dataset_name = "example1_n=100_m=50"  #"example1"

# Set the model parameters
dim = 2
order = 2
epochs_num = 100
learning_rate = 0.1
samples_num = 10
batch_size = 10
seed = 123

# Load the dataset
data = load(name=dataset_name, dataset_type="syn")
# Normalize time list
data['times'] = [value / float(max(data['times'])) for value in data['times']]

data['events'], data['col_indices'], data['times'] = batching(event_mat=data['events'], event_times=data['times'], batch_size=batch_size)

# Define the model
p2v = Pol2VecBinary(train_data=data, dim=dim, order=order, epochs_num=epochs_num,
                    learning_rate=learning_rate, samples_num=samples_num, seed=seed)
# Train the model
train_loss_list, test_loss_list = p2v.learn(type="seq")

# Save the training loss
plt.figure()
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Negative Log Likelihood")
plt.plot(train_loss_list, 'k.-')
plt.show()