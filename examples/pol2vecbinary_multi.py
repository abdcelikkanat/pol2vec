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
torch.set_num_threads(16)


def batching(event_mat, event_times, batch_size, max_time):
    assert batch_size <= max_time, "Batch size must be smaller than max time."
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
suffix = "deneme" #"test"

# Set the model parameters
dim = 2
order = 2
metric = "mah"
metric_param = "diag"
epochs_num = 100
learning_rate = 0.1
batch_size = 5
max_column = 10
sigma_grad = False
seed = 123
verbose = True
learn = True

# Define the file name
filename = "{}_{}_order={}_{}_{}_epochs={}_lr={}_batch_size={}_max_column={}_seed={}".format(dataset_name, suffix, order, metric, metric_param, epochs_num, learning_rate, batch_size, max_column, seed)

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
data['events'] = data['events']

# Batching
train_data = dict()
train_data['events'], train_data['col_indices'], train_data['times'] = batching(event_mat=data['events'], event_times=data['times'],
                                                                                batch_size=batch_size, max_time=max_column)
# Define the model
p2v = Pol2VecMulti(train_data=train_data, dim=dim, order=order, metric=metric, metric_param=metric_param,
                   epochs_num=epochs_num, learning_rate=learning_rate, sigma_grad=sigma_grad, seed=seed, verbose=verbose)
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

    print("Completed.")
else:
    p2v.load_state_dict(torch.load(model_file_path))

    # Set the file path for the animation
    animation_file_path = os.path.join(figures_folder, "{}_animation.gif".format(filename))

    # Sample some time points for the visualization
    time_points_num = max_column
    time_points = np.linspace(0, 1, time_points_num)

    # Get the column times list in the chosen interval
    col_times = data['times'][:max_column]
    # Get the non-zero indices in the chosen interval
    mat_indices = torch.tensor(data['events'][:, :max_column].todense(), dtype=torch.int8).nonzero()
    # Get the number of rows in the chosen interval
    number_of_rows = max(mat_indices.T[0])+1
    # Construct the estimated embedding tensor which is of size T x N x D
    z_est = np.zeros(shape=(time_points_num, number_of_rows, dim), dtype=np.float)
    # Get the estimated positions
    z_est[mat_indices.T[1], mat_indices.T[0], :] = p2v.get_all_row_positions(mat_indices=mat_indices, col_times=col_times).detach().numpy()

    # Perform the animation
    color_mat = []
    for t_idx in range(time_points_num):
        current_colors = []
        for node in range(data['events'].shape[0]):
            if data['events'][node, t_idx] == 1:  # no
                current_colors.append('maroon')
            elif data['events'][node, t_idx] == 2:  # yes
                current_colors.append('navy')
            elif data['events'][node, t_idx] == 0:
                current_colors.append('white')
            else:
                current_colors.append('yellowgreen')
        color_mat.append(current_colors)

    anim = Animation(timePoints=time_points, r=1, c=1, figSize=(8, 6), bgColor='white', color=color_mat,
                     marker='.', markerSize=10, delay=500, margin=[0.1, 0.1], label=False)
    anim.addData(x=z_est, index=0, canvas=None,  title="Estimation")
    anim.plot(filePath=animation_file_path)


    # Assignment
    y_assignment = p2v.get_label_assignments(mat_indices=mat_indices, col_times=data['times'][:max_column])
    # print("Second")
    # # Set the file path for the animation
    # animation_file_path2 = os.path.join(figures_folder, "{}_animation2.gif".format(filename))
    # color_mat = []
    # for t_idx in range(time_points_num):
    #     current_colors = []
    #     for node in range(data['events'].shape[0]):
    #         current_colors.append('white')
    #     color_mat.append(current_colors)
    #
    # for idx, mat_idx in enumerate(mat_indices):
    #     if y_assignment[idx] == 0:
    #         color_mat[mat_idx[1]][mat_idx[0]] = 'r'  # no
    #     elif y_assignment[idx] == 1:  # yes
    #         color_mat[mat_idx[1]][mat_idx[0]] = 'b'
    #     elif y_assignment[idx] == 2:  # yes
    #         color_mat[mat_idx[1]][mat_idx[0]] = 'green'
    #
    # anim = Animation(timePoints=time_points, r=1, c=1, figSize=(8, 6), bgColor='white', color=color_mat,
    #                  marker='.', markerSize=3, delay=1000, margin=[0.1, 10], label=False)
    # anim.addData(x=z_est, index=0, title="Estimation")
    # anim.plot(filePath=animation_file_path2)

    # --


    # values = p2v.compute_all_f(mat_indices=mat_indices, col_times=col_times)
    # true labels
    y_true = []
    for row_idx, col_idx in mat_indices:
        y_true.append( [ data['events'][row_idx, col_idx] - 1 ] )
    # estimated labels
    y_pred = [[l] for l in y_assignment]

    K = len(set([y for yy in y_true for y in yy]))

    # # Find the predictions, each node can have multiple labels
    # test_prob = np.asarray(ovr.predict_proba(test_features))
    # y_pred = []
    # for i in range(test_labels.shape[0]):
    #     k = test_labels[i].getnnz()  # The number of labels to be predicted
    #     pred = test_prob[i, :].argsort()[-k:]
    #     y_pred.append(pred)
    #
    # Find the true labels
    # y_true = [[] for _ in range(test_labels.shape[0])]
    # co = test_labels.tocoo()
    # for i, j in zip(co.row, co.col):
    #     y_true[i].append(j)

    score_types = ['micro', 'macro']
    train_ratios = [0.1, 0.5, 0.9]
    results = {score_type: {train_ratio: 0. for train_ratio in train_ratios} for score_type in score_types}
    for train_ratio in train_ratios:

        mlb = MultiLabelBinarizer(range(K))
        for score_t in score_types:
            score = f1_score(y_true=mlb.fit_transform(y_true),
                             y_pred=mlb.fit_transform(y_pred),
                             average=score_t)

            results[score_t][train_ratio] = score

    for score_t in score_types:
        print(score_t, [results[score_t][train_ratio] for train_ratio in train_ratios])

    # Simon's baseline
    print("Simon's baseline")
    max_row_idx = max(mat_indices.T[0])+1
    label_counts = np.zeros(shape=(max_row_idx, K+1), dtype=np.int)
    for row_idx, col_idx in mat_indices:
        label_counts[row_idx, data['events'][row_idx, col_idx]] += 1

    # estimated labels
    y_pred = [[label_counts[row_idx, 1:].argmax()] for row_idx, col_idx in mat_indices]

    results = {score_type: {train_ratio: 0. for train_ratio in train_ratios} for score_type in score_types}
    for train_ratio in train_ratios:

        mlb = MultiLabelBinarizer(range(K))
        for score_t in score_types:
            score = f1_score(y_true=mlb.fit_transform(y_true),
                             y_pred=mlb.fit_transform(y_pred),
                             average=score_t)

            results[score_t][train_ratio] = score

    for score_t in score_types:
        print(score_t, [results[score_t][train_ratio] for train_ratio in train_ratios])


    print(p2v.get_variable_at(i=0, order_index=0, t=0.5) )
    print(p2v.get_variable_at(i=0, order_index=1, t=0.5))
    print(p2v.get_variable_at(i=0, order_index=2, t=0.5) )
    print(p2v.get_variable_at(i=0, order_index=2, t=0.5) )
    # print(p2v.get_all_row_positions(mat_indices=, col_times=[0.5]) )