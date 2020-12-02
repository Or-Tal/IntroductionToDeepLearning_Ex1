import argparse
from models import *
from gen_dset_from_txt_files import gen_encoded_vector, DATA_X, DATA_Y
import numpy as np
import logging

# --------------- Constants --------------

AMINO_MAP = {acid: i for i, acid in enumerate(['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                                               'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'])}
SEQ_LEN = 9
BEST_MODELS = ["mlp_v2_o1", "mlp_v2_o1", "mlp_v2_o1", "mlp_v2_o1", "mlp_v0_o1"]
MATCHING_BEST_CONFIGURATIONS = [1, 8, 3, 4, 3]
NUM_OUTPUTS = [1, 1, 1, 1, 1]


# ------------ Main functions ------------

def detect_covid_in_fasta(fasta_file_path):
    """
    main function for detecting covid sequences in fasta file
    :return: starting idxs list of positive covid detected 9-mer sequences
    """
    # break file into encoded matrix
    encoded_mat, non_encoded_arr = generate_encoded_matrix_from_fasta(fasta_file_path)

    # gen committee method
    committee = build_committee_model()

    # predict
    predictions, probabilities = committee(encoded_mat)
    return  predictions, probabilities, non_encoded_arr


# ----------- Helper functions -----------


def generate_encoded_matrix_from_fasta(fasta_file_path):
    """
    this function generates an encoded matrix of examples from .fasta file
    """
    # read .fasta file and discard header
    with open(fasta_file_path, "r") as f:
        lines = f.read().split("\n")
        if lines[0][0] == '>':
            lines = lines[1:]
        amino_sequence = "".join(lines)

    # loop over k-mer segments
    k_mer_examples, k_mer = list(), list()
    for i in np.arange(len(amino_sequence) - SEQ_LEN):
        k_mer_examples.append(gen_encoded_vector(amino_sequence[i:i + 9], AMINO_MAP))
        k_mer.append(amino_sequence[i:i + 9])

    return np.array(k_mer_examples), np.array(k_mer)


def build_committee_model():
    """
    this function builds the committee model from several pretrained MLP models
    :return: committee model object
    """
    models = list()
    for i in range(len(BEST_MODELS)):
        base_dir = f"./weights_dir/{BEST_MODELS[i]}"
        configurations = np.load(f"{base_dir}/configurations.npy", allow_pickle=True).tolist()
        cfg = configurations[MATCHING_BEST_CONFIGURATIONS[i]]
        n_output = NUM_OUTPUTS[i]
        model = MLP(cfg[0], cfg[1], n_output)
        model.load_weights(f"{base_dir}/{BEST_MODELS[i]}_{MATCHING_BEST_CONFIGURATIONS[i]}")
        models.append(model)

    return MLP_COMMITTEE(models)


def angular_loss(y_true, y_pred):
    """
    function to calculate mean absolute loss between angles
    :param y_true: vector of ground truth values, assuming values in range [0,360] (including floats)
    :param y_pred: vector of predictions, would be taken as modulus 360 angle
    :return: mean absolute loss between angles
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    shift = (360 - y_true) % 360  # calculate a shift vector
    y_pred = (y_pred + shift) % 360
    loss = tf.where(y_pred < 180, y_pred, 360 - y_pred)
    return tf.reduce_mean(loss)


def angular_loss2(y_true, y_pred):
    """
    function to calculate mean absolute loss between angles
    :param y_true: vector of ground truth values
    :param y_pred: vector of predictions
    :return: mean absolute loss between angles
    """
    y_hat = y_true + 180 - ((y_true + 180) // 360) * 360  # not to limit ourselves to integers
    y_pred = y_pred % 360
    loss = tf.abs(tf.where(y_pred < y_hat, y_pred + 360 - y_true, y_true - y_pred))
    return tf.reduce_mean(loss)



if __name__ == "__main__":
    tf.get_logger().setLevel(logging.ERROR)

    # parse arguments
    parser = argparse.ArgumentParser(description='find covid sequences in .fasta file')
    parser.add_argument('--dst', required=True, help='Path to target file to classify', nargs=1)

    # test committee on test_set
    d = np.load("./test_dset.npy", allow_pickle=True).tolist()
    committee = build_committee_model()
    predictions, probabilities = committee(d[DATA_X])
    print("test:")
    for i in range(len(probabilities)):
        print(f"prediction: {predictions[i]}, g.t: {d[DATA_Y][i]}, probability: {probabilities[i]}")
    same = np.sum(predictions * d[DATA_Y].flatten()) + np.sum((1 - predictions) * (1 - d[DATA_Y].flatten()))
    print(f"accuracy = {same / len(predictions)}")

    # detect
    predictions, probabilities, nine_mer = detect_covid_in_fasta(parser.parse_args().dst[0])

    # print out results
    print("\nAll positive classified 9-mers: "
          "(index in .fasta file), (9-mer matching that index), (class_probabilities)")
    out_lst = [(i, nine_mer[i], probabilities[i]) for i in np.nonzero(predictions)[0]]
    out_lst = sorted(out_lst, key=lambda x: -x[2])
    for i, n, p in out_lst:
        print(f"index: {i}, 9-mer: {nine_mer[i]}, probability: {probabilities[i]}")
