import pickle

def read_pickle(
    pickle_file_path: str,
) -> dict:
    """Read a pickled file and return a dict

    Args:
        pickle_file_path (str): the file path of pickled file

    Returns:
        dict: the content of pickled file
    """
    with open(pickle_file_path, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict


def save_pickle(
    obj,
    save_to_file: str,
):
    """Save an obj to a pickle file.

    Args:
        obj (_type_): the obj to be saved
        save_to_file (str): the file name (with or with path) for saving obj
    """
    assert save_to_file[-4:] == ".pkl", "please provide file name ends with .pkl"        
    o = open(save_to_file, 'wb')
    pickle.dump(obj, o)
    o.close()