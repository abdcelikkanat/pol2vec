import csv
from scipy.sparse import csr_matrix

cast_codes = []
member_ids = []
bill_ids = []

with open('./congress_links.csv') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        cast_codes.append(int(row[0]))
        member_ids.append(int(row[1]))
        bill_ids.append(int(row[2]))

num_of_cast_codes = len(set(cast_codes))
num_of_member_ids = len(set(member_ids))
num_of_bill_ids = len(set(bill_ids))

print(set(cast_codes))
print("Number of unique cast codes: {}/{}".format(num_of_cast_codes, len(cast_codes)))
print("Number of unique member ids: {}/{}".format(num_of_member_ids, len(member_ids)))
print("Number of unique bill ids: {}/{}".format(num_of_bill_ids, len(bill_ids)))


M = csr_matrix((cast_codes, (member_ids, bill_ids)), shape=(num_of_member_ids, num_of_bill_ids))

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.figure(figsize=(360, 50))
cmap = ListedColormap(['r', 'b', 'y'])
plt.matshow(M.todense(), cmap=cmap)
plt.show()