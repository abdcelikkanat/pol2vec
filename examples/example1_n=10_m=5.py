import os
import matplotlib.pyplot as plt
from models.pol2vec import *
from datasets.loader import *

# Define the global variables
script_folder = os.path.dirname(__file__)
base_folder = os.path.realpath(os.path.join(script_folder, ".."))
dataset_folder = os.path.realpath(os.path.join(base_folder, "datasets"))

# Set the dataset name
dataset_name = "example1"

# Set the model parameters
dim = 2
order = 2
epochs_num = 100
learning_rate = 0.1
samples_num = 10
seed = 123

# Load the dataset
data = load(name="example1", dataset_type="syn")
data['times'] = [value / 10. for value in data['times']]

# Define the model
p2v = Pol2Vec(train_data=data, dim=dim, order=order, epochs_num=epochs_num,
              learning_rate=learning_rate, samples_num=samples_num, seed=seed)
# Train the model
train_loss_list, test_loss_list = p2v.learn(type="seq")

plt.figure()
plt.plot(train_loss_list, 'r.')
plt.show()