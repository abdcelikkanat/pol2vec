import os
import pickle as pkl

dir_path = os.path.dirname(os.path.realpath(__file__))


def load(name, dataset_type="real"):

    if dataset_type == "synthetic" or dataset_type == "syn" or dataset_type == "artificial":

        file_path = os.path.join(dir_path, "synthetic", name, name + '.pickle')

        with open(file_path, 'rb') as f:
            events = pkl.load(f)

    elif dataset_type == "real":

        if type(name) is str:

            file_path = os.path.join(dir_path, "real", name, name+'.pickle')

            with open(file_path, 'rb') as f:
                events = pkl.load(f)

        else:
            raise ValueError("Wrong dataset name!")

    else:

        raise ValueError("Invalid dataset type name!")

    return events

