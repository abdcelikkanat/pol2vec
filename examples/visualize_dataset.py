import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import matplotlib.pyplot as plt
from models.pol2vecmulti import *
from datasets.loader import *
from utilities.animation import *
from utilities.complete_motion import CompleteMotion
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from matplotlib.colors import ListedColormap
torch.set_num_threads(16)


def batching(event_mat, event_times, batch_size, max_time):
    event_mat_list = []
    event_times_list = []
    col_indices_list = []

    # for i in range(0, event_mat.shape[1], batch_size):
    for i in range(0, max_time, batch_size):

        event_mat_list.append(event_mat[:, i:i + batch_size])
        col_indices_list.append(list(range(i, i + batch_size)))
        event_times_list.append(event_times[i:i + batch_size])

    return event_mat_list, col_indices_list, event_times_list


# Define the global variables
script_folder = os.path.dirname(__file__)
base_folder = os.path.realpath(os.path.join(script_folder, ".."))
dataset_folder = os.path.realpath(os.path.join(base_folder, "datasets"))

# Set the dataset name
dataset_name = "congress" #"arxiv_19-20"  #"example1
suffix = "multi"

# Set the model parameters
dim = 2
order = 2
epochs_num = 100
learning_rate = 0.1
samples_num = 10
batch_size = 50
max_time = 100
seed = 123
learn = False

# Define the file name
filename = "{}_{}_epochs={}_lr={}_batch_size={}_seed={}".format(dataset_name, suffix, epochs_num, learning_rate, batch_size, seed)

# Set the figures folder
figures_folder = os.path.join(base_folder, "figures", filename)

# Create the folder
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder)

# Define the path for the model parameters
model_file_path = os.path.join(figures_folder, filename+".model")

# Define the file path for the training loss
loss_file_path = os.path.join(figures_folder, "{}_train_loss.png".format(filename))

# Load the dataset
data = load(name=dataset_name, dataset_type="real")

# Normalize time list
max_el = float(max(data['times']))
data['times'] = [value / max_el for value in data['times']]

# Concert the data to dense format
data['events'] = data['events'] #.todense()

mat = []
row_indices, col_indices = data['events'][:, :max_time].nonzero()

# mat = np.zeros(shape=(max(row_indices), max(col_indices)), )

#
plt.figure()
cmap = ListedColormap(['w', 'darkred', 'lightskyblue', 'gray'])
plt.matshow(data['events'][:max(row_indices), :max_time].todense(), cmap=cmap)
plt.xlabel("Bills")
plt.ylabel("Politicians")
plt.show()