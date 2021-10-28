import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
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
dataset_name = "arxiv_19-20" #"arxiv_19-20"  #"example1
suffix = "example1"

# Set the model parameters
dim = 2
order = 2
epochs_num = 100
learning_rate = 0.1
samples_num = 10
batch_size = 5
seed = 123
learn = True

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

print(data['events'].shape)

data['events'], data['col_indices'], data['times'] = batching(event_mat=data['events'], event_times=data['times'], batch_size=batch_size)



# Define the model
p2v = Pol2VecBinary(train_data=data, dim=dim, order=order, epochs_num=epochs_num,
                    learning_rate=learning_rate, samples_num=samples_num, seed=seed)

if learn:
    print("Learn")
    # Train the model
    train_loss_list, test_loss_list = p2v.learn(type="seq")

    # Save the model
    torch.save(p2v.state_dict(), model_file_path)

    # Save the training loss
    plt.figure()
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Negative Log Likelihood")
    plt.plot(train_loss_list, 'k.-')
    plt.savefig(loss_file_path)

    p2v.load_state_dict(torch.load(model_file_path))

    print("Completed.")
else:
    pass



# # Set the file path for the animation
# animation_file_path = os.path.join(base_folder, "figures", "{}_animation.gif".format(filename_name))
#
# # Sample some time points for the visualization
# time_points_num = data['events'].shape[1]
# time_points = np.linspace(0, 1, time_points_num)
#
# # Get the correct latent positions
# cm = CompleteMotion(data['z_p'], data['times'])
# z_true = np.zeros(shape=(time_points_num, data['events'].shape[0], dim), dtype=np.float)
# for t_idx, t in enumerate(time_points):
#     z_true[t_idx, :, :] = cm.get_current_positions(t=t)
#
# # Get the estimated positions
# z_est = np.zeros(shape=(time_points_num, data['events'].shape[0], dim), dtype=np.float)
# for t_idx, t in enumerate(time_points):
#     z_est[t_idx, :, :] = p2v.get_all_politician_positions(t=t).detach().numpy()
#
# # Perform the animation
# anim = Animation(timePoints=time_points,
#                          r=1, c=2, figSize=(16, 6), bgColor='white',
#                          # color='k', #
#                         color=[['r' if data['events'][i, bill_idx] else 'k' for i in range(data['events'].shape[0])] for bill_idx in range(data['events'].shape[1])],
#                          marker='.', markerSize=10, delay=1000, margin=[0.1, 10], label=False)
# anim.addData(x=z_true, index=0, title="True")
# anim.addData(x=z_est, index=1, title="Estimation")
# anim.plot(filePath=animation_file_path)