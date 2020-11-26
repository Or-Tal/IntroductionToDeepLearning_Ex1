import numpy as np
import argparse
import os

# --------------- Constants --------------
DATA_X = 'X'
DATA_Y = 'Y'
DATA_X_P = 'X_P'
DATA_Y_P = 'Y_P'
DATA_X_N = 'X_N'
DATA_Y_N = 'Y_N'
TRAIN = 'train_dset.npy'
VAL = 'val_dset.npy'
TEST = 'test_dset.npy'
TRAIN_v2 = 'train_dset_v2.npy'
TEST_v2 = 'test_dset_v2.npy'
NUM_PROTEINS = 20

# ------------ Main functions ------------


def gen_npy_data(pos_example_path, neg_example_path, save_dir, test_p=0.05):
    """
    this function takes in two txt files: one for positive examples, and one for negative examples
    and saves a .npy of the whole dataset where values are encoded as one hot matrix and binary vector
    :param pos_example_path: path to the positive examples dataset
    :param neg_example_path: path to the negative examples dataset
    :param save_dir: path to the directory to store the dataset .npy files in
    :param test_p: what percentage of the data should the test set be p in range(0,1)
    """
    assert 0 < test_p < 1, "invalid test_p was given"
    tmp_x_p, tmp_y_p, tmp_y2_p = list(), list(), list()
    tmp_x_n, tmp_y_n, tmp_y2_n = list(), list(), list()

    # read pos file and append it to the dataset
    with open(pos_example_path, 'r') as f:
        for line in f.readlines():
            t = line[:-1] if line[-1] == '\n' else line
            tmp_x_p.append(t)
            tmp_y_p.append(1.)
            tmp_y2_p.append([1., 0.])

    # read neg file and append it to the dataset
    with open(neg_example_path, 'r') as f:
        for line in f.readlines():
            t = line[:-1] if line[-1] == '\n' else line
            tmp_x_n.append(t)
            tmp_y_n.append(0.)
            tmp_y2_n.append([0., 1.])

    # get embeddings
    tmp_x_p = _str_embedding(tmp_x_p)
    tmp_y_p = np.array(tmp_y_p)
    tmp_y2_p = np.array(tmp_y2_p)
    tmp_x_n = _str_embedding(tmp_x_n)
    tmp_y_n = np.array(tmp_y_n)
    tmp_y2_n = np.array(tmp_y2_n)
    tmp_y_p, tmp_y_n = tmp_y_p.reshape((tmp_y_p.shape[0], 1)), tmp_y_n.reshape((tmp_y_n.shape[0], 1))

    # split to train/test sets - match num positive to num negative examples
    train, test, train2, test2 = gen_train_test(tmp_x_p, tmp_x_n, tmp_y_p, tmp_y_n, tmp_y2_p, tmp_y2_n, test_p)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    np.save(f"{save_dir}/{TRAIN}", train, allow_pickle=True)
    np.save(f"{save_dir}/{TEST}", test, allow_pickle=True)
    np.save(f"{save_dir}/{TRAIN_v2}", train2, allow_pickle=True)
    np.save(f"{save_dir}/{TEST_v2}", test2, allow_pickle=True)


# ----------- Helper functions -----------

def gen_train_test(tmp_x_p, tmp_x_n, tmp_y_p, tmp_y_n, tmp_y2_p, tmp_y2_n, test_p):
    """
    split to train/val/test
    """
    # random order positive indices
    N = len(tmp_x_p)
    idxs = np.arange(N)
    np.random.shuffle(idxs)

    # random order negative indices
    N2 = len(tmp_x_n)
    idxs2 = np.arange(N2)
    np.random.shuffle(idxs2)

    # split into train, test sets (train2/test2 = double label output)
    t = int((test_p) * N)
    test_x_p, test_y_p = tmp_x_p[idxs[:t]], tmp_y_p[idxs[:t]]
    train_y2_p, test_y2_p = tmp_y2_p[idxs[t:]], tmp_y2_p[idxs[:t]]
    train_x_p, train_y_p = tmp_x_p[idxs[t:]], tmp_y_p[idxs[t:]]
    test_x_n, test_y_n = tmp_x_n[idxs2[:t]], tmp_y_n[idxs2[:t]]
    train_y2_n, test_y2_n = tmp_y2_n[idxs2[t:]], tmp_y2_n[idxs2[:t]]
    train_x_n, train_y_n = tmp_x_n[idxs2[t:]], tmp_y_n[idxs2[t:]]
    test_x, test_y, test_y2 = np.concatenate([test_x_p, test_x_n], axis=0), \
                              np.concatenate([test_y_p, test_y_n], axis=0), \
                              np.concatenate([test_y2_p, test_y2_n], axis=0)

    # generate data dictionaries
    train = {DATA_X_P: train_x_p, DATA_Y_P: train_y_p, DATA_X_N: train_x_n, DATA_Y_N: train_y_n}
    train2 = {DATA_X_P: train_x_p, DATA_Y_P: train_y2_p, DATA_X_N: train_x_n, DATA_Y_N: train_y2_n}
    test = {DATA_X: test_x, DATA_Y: test_y}
    test2 = {DATA_X: test_x, DATA_Y: test_y2}
    return train, test, train2, test2


def gen_amino_acid_map():
    """
    There are 20 amino acid types:
        alanine - ala - A
        arginine - arg - R
        asparagine - asn - N
        aspartic acid - asp - D
        cysteine - cys - C
        glutamine - gln - Q
        glutamic acid - glu - E
        glycine - gly - G
        histidine - his - H
        isoleucine - ile - I
        leucine - leu - L
        lysine - lys - K
        methionine - met - M
        phenylalanine - phe - F
        proline - pro - P
        serine - ser - S
        threonine - thr - T
        tryptophan - trp - W
        tyrosine - tyr - Y
        valine - val - V
    this function returns a mapping matching these types
    """
    amino_map = {acid: i for i, acid in enumerate(['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                                                   'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'])}
    return amino_map


def gen_encoded_vector(str, amino_map):
    """
    this function creats an encoded vector from amino_map representation
    :param str: string to encode
    :param amino_map: char to idx mapping
    :return: encoded vector
    """
    out_vec = np.zeros((len(str), NUM_PROTEINS))
    for i, char in enumerate(str.upper()):
        out_vec[i, amino_map[char]] = 1
    return out_vec.flatten()


def _str_embedding(str_list):
    """
    this function receives a list of n strings of length m (n,m) and converts them to (n, 26 * m) one_hot matrix
    :param str_list:
    :return:
    """

    # init one_hot_matrix
    n, m = len(str_list), len(str_list[0]) if str_list[0][-1] != "\n" else len(str_list[0]) - 1
    one_hot_matrix = np.zeros((n, NUM_PROTEINS * m))
    amino_map = gen_amino_acid_map()  # generate char mapping

    # fill one hot matrix
    for i in range(n):
        one_hot_matrix[i] = gen_encoded_vector(str_list[i], amino_map)

    return one_hot_matrix


def _parse_args():
    """
    function for parsing input arguments
    """
    parser = argparse.ArgumentParser(description='Encode positive and negative examples into dataset')
    parser.add_argument('--neg', required=True, help='Path to negative examples .txt file', nargs=1)
    parser.add_argument('--pos', required=True, help='Path to positive examples .txt file', nargs=1)
    parser.add_argument('--out_dir', required=True, help='Full path to output to save the dataset to', nargs=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    gen_npy_data(args.pos[0], args.neg[0], args.out_dir[0])