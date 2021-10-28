import os
import scipy
import pickle
from scipy.sparse import csr_matrix


cast_codes = []
member_ids = []
bill_ids = []


dataset_name = "arxiv_19-20"

folder_path = "."
events = scipy.sparse.load_npz(os.path.join(folder_path, '{}.npz'.format(dataset_name)))
event_times = range(events.shape[1])



# M = csr_matrix((cast_codes, (member_ids, bill_ids)), shape=(num_of_member_ids, num_of_bill_ids))

# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# plt.figure(figsize=(36, 24))
# cmap = ListedColormap(['k', 'w'])
# plt.matshow(events.todense(), cmap=cmap)
# plt.matshow(events, cmap="gray_r")
# plt.show()


filename = "{}".format(dataset_name)

file_path = os.path.join(folder_path, "{}.pickle".format(filename))
with open(file_path, 'wb') as f:
    pickle.dump({'events': events, 'times': event_times}, f, protocol=pickle.HIGHEST_PROTOCOL)