import os
import matplotlib.pyplot as plt
from models.pol2vec import *
from datasets.loader import *
from utilities.animation import *
from utilities.complete_motion import CompleteMotion
torch.set_num_threads(16)


# Define the global variables
script_folder = os.path.dirname(__file__)
base_folder = os.path.realpath(os.path.join(script_folder, ".."))
dataset_folder = os.path.realpath(os.path.join(base_folder, "datasets"))

# Set the dataset name
dataset_name = "example1_n=10_m=5"  #"example1"

# Set the model parameters
dim = 2
order = 2
epochs_num = 50
learning_rate = 0.1
samples_num = 10
seed = 123

# Load the dataset
data = load(name=dataset_name, dataset_type="syn")
# Normalize time list
data['times'] = [value / float(max(data['times'])) for value in data['times']]

# Define the model
p2v = Pol2Vec(train_data=data, dim=dim, order=order, epochs_num=epochs_num,
              learning_rate=learning_rate, samples_num=samples_num, seed=seed)
# Train the model
train_loss_list, test_loss_list = p2v.learn(type="seq")


# Define the file name
filename_name = "{}_epochs={}_lr={}_seed={}".format(dataset_name, epochs_num, learning_rate, seed)

# Define the path for the model parameters
model_file_path = os.path.join(script_folder, filename_name+".model")
torch.save(p2v.state_dict(), model_file_path)
# p2v.load_state_dict(torch.load(model_file_path))

# Save the training loss
loss_file_path = os.path.join(base_folder, "figures", "{}_train_loss.png".format(filename_name))
plt.figure()
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Negative Log Likelihood")
plt.plot(train_loss_list, 'k.-')
plt.savefig(loss_file_path)

# Set the file path for the animation
animation_file_path = os.path.join(base_folder, "figures", "{}_animation.gif".format(filename_name))

# Sample some time points for the visualization
time_points_num = data['events'].shape[1]
time_points = np.linspace(0, 1, time_points_num)

# Get the correct latent positions
cm = CompleteMotion(data['z_p'], data['times'])
z_true = np.zeros(shape=(time_points_num, data['events'].shape[0], dim), dtype=np.float)
for t_idx, t in enumerate(time_points):
    z_true[t_idx, :, :] = cm.get_current_positions(t=t)

# Get the estimated positions
z_est = np.zeros(shape=(time_points_num, data['events'].shape[0], dim), dtype=np.float)
for t_idx, t in enumerate(time_points):
    z_est[t_idx, :, :] = p2v.get_all_politician_positions(t=t).detach().numpy()

# Perform the animation
anim = Animation(timePoints=time_points,
                         r=1, c=2, figSize=(16, 6), bgColor='white',
                         # color='k', #
                        color=[['r' if data['events'][i, bill_idx] else 'k' for i in range(data['events'].shape[0])] for bill_idx in range(data['events'].shape[1])],
                         marker='.', markerSize=10, delay=1000, margin=[0.1, 10], label=False)
anim.addData(x=z_true, index=0, title="True")
anim.addData(x=z_est, index=1, title="Estimation")
anim.plot(filePath=animation_file_path)