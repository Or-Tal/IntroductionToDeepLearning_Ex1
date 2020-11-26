import argparse
import os
import datetime
from models import *
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint
from gen_dset_from_txt_files import DATA_X_P, DATA_X_N, DATA_Y_P, DATA_Y_N, DATA_X, DATA_Y
from data_generator import *
import logging

# --------------- Constants --------------

SUMMARY = 'train_summary'
EVAL = 'evaluation_summary'
CONFIGURATIONS = 'configurations.npy'

# ------------ Main functions ------------


def train(model, train_data, val_data, n_epochs, lr, save_w_path, model_name,
          metrics, batch_size=128, monitor=None, num_outputs=2):
    """
    main training function
    """

    # define generators that samples equal number of positive and negative samples in each batch
    train_gen = DataGenerator(train_data, batch_size)

    # define callbacks
    callbacks = list()
    if monitor is None:
        callbacks.append(ModelCheckpoint(f"{save_w_path}/{model_name}", save_best_only=True,
                                         save_weights_only=True, verbose=1))
    else:
        callbacks.append(ModelCheckpoint(f"{save_w_path}/{model_name}", save_best_only=True,
                                         save_weights_only=True, monitor=monitor, verbose=1,
                                         mode='max'))

    log_dir = f"{save_w_path}/logs/{model_name}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))

    # define the optimizer, loss and compile model
    loss = tf.keras.losses.BinaryCrossentropy() if num_outputs == 2 else \
        tf.keras.losses.MeanAbsoluteError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # fit
    history = model.fit(x=train_gen, batch_size=batch_size, epochs=n_epochs,
                        validation_data=val_data, callbacks=callbacks)
    return history, loss


def train_loop(base_w_dir, train_dset_path, test_dset_path, model_name, metrics,
               model_cls, monitor, num_outputs=2):
    """
    training loop - iterates over various configurations
    :return:
    """
    print("begin train loop")
    gen_all_dirs(base_w_dir, model_name)

    # gen training vars
    loop_vars = gen_loop_vars()
    train_data, val_data = gen_train_val_sets(np.load(train_dset_path, allow_pickle=True).tolist())

    # save configurations file
    configurations = {i: vars for i, vars in enumerate(loop_vars)}
    np.save(f"{base_w_dir}/{model_name}/{CONFIGURATIONS}", configurations)

    # train loop
    loss_out = None
    for i, vars in enumerate(loop_vars):
        layers, activations, n_epochs, lr = vars
        # build model and train
        model = model_cls(layers, activations, num_outputs)
        history, loss_func = train(model=model, train_data=train_data, n_epochs=n_epochs, lr=lr,
                                   save_w_path=f"{base_w_dir}/{model_name}",
                                   model_name=f"{model_name}_{i}", metrics=metrics,
                                   val_data=val_data, monitor=monitor, num_outputs=num_outputs)
        if loss_out is None:
            loss_out = loss_func

        # log for monitoring
        with open(f"{base_w_dir}/{model_name}/{SUMMARY}", "a") as f:
            f.write("-------------\n"
                    "Model:               {}\n"
                    "Validation Loss:     {}\n"
                    "Validation Accuracy: {}\n"
                    "Validation Precision:{}\n"
                    "Validation Recall:   {}\n".format(model_name,
                                                       np.mean(history.history['val_loss']),
                                                       np.mean(history.history["val_acc"]),
                                                       np.mean(history.history["val_precision"]),
                                                       np.mean(history.history["val_recall"])))

    # eval loop
    test_data = np.load(test_dset_path, allow_pickle=True).tolist()
    x_test, y_test = test_data[DATA_X], test_data[DATA_Y]
    with open(f"{base_w_dir}/{model_name}/{EVAL}", "w") as f:
        for i in configurations.keys():
            layers, activations, n_epochs, lr = configurations[i]

            # build model and evaluate
            model = model_cls(layers, activations, num_outputs)
            model.compile(loss=tf.keras.losses.BinaryCrossentropy(), metrics=metrics)
            model.load_weights(f"{base_w_dir}/{model_name}/{model_name}_{i}")
            results = model.evaluate(x_test, y_test)

            # log for monitoring
            f.write(f"model: {model_name}_{i}, Loss: {results[0]}, Acc: {results[1]}, "
                    f"Prec: {results[2]}, Rec: {results[3]}\n")
    print("==== DONE ====")


# ----------- Helper functions -----------


def gen_loop_vars():
    """
    this function produces train loop vars
    """
    n_layers = [5, 10, 20]
    n_nodes = [100, 300, 500]  # num nodes per layer
    n_epochs = [100]
    lrs = [1e-4, 1e-3]
    train_vars = list()
    for i, n in enumerate(n_layers):
        for m in n_nodes:
            for e in n_epochs:
                for lr in lrs:
                    layers = [m] * (n - 1) + [m // 2]
                    train_vars.append((layers, ['relu'] * n, e, lr))
    return train_vars


def gen_all_dirs(base_dir, model_name):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    if not os.path.exists(f"{base_dir}/{model_name}"):
        os.mkdir(f"{base_dir}/{model_name}")
    if not os.path.exists(f"{base_dir}/{model_name}/logs"):
        os.mkdir(f"{base_dir}/{model_name}/logs")


def parse_args():
    """
    function for parsing input arguments
    """
    parser = argparse.ArgumentParser(description='Encode positive and negative examples into dataset')
    parser.add_argument('--train', required=True, help='Path to train_dset.npy file', nargs=1)
    parser.add_argument('--test', required=True, help='Path to test_dset.npy file', nargs=1)
    parser.add_argument('--w_dir', required=True, help='path to weights dir', nargs=1)
    parser.add_argument('--name', required=False, help='model name', nargs=1)
    return parser.parse_args()


def gen_train_val_sets(data_dict, val_p=0.1/0.95):
    """
    this function randomly produces a validation set for each training loop.
    :param data_dict: preprocessed train dictionary
    :param val_p: relative percentage from the train set to be used as validation set
    :return:
    """
    # gen pos/neg sets
    x_p, x_n, y_p, y_n = data_dict[DATA_X_P], data_dict[DATA_X_N], data_dict[DATA_Y_P], \
                         data_dict[DATA_Y_N]

    # shuffle
    idxs_p, idxs_n = np.arange(x_p.shape[0]), np.arange(x_n.shape[0])
    np.random.shuffle(idxs_p), np.random.shuffle(idxs_n)
    x_p, y_p, x_n, y_n = x_p[idxs_p], y_p[idxs_p], x_n[idxs_n], y_n[idxs_n]

    # produce train, val sets
    val_data = (np.concatenate([x_n[:int(val_p * x_n.shape[0])],
                                x_p[:int(val_p * x_p.shape[0])]], axis=0),
                np.concatenate([y_n[:int(val_p * y_n.shape[0])],
                                y_p[:int(val_p * y_p.shape[0])]], axis=0))
    train_data = {
        DATA_X_P: x_p[int(val_p * x_p.shape[0]):], DATA_X_N: x_n[int(val_p * x_n.shape[0]):],
        DATA_Y_P: y_p[int(val_p * y_p.shape[0]):], DATA_Y_N: y_n[int(val_p * y_n.shape[0]):]
    }
    return train_data, val_data


if __name__ == "__main__":

    # parse input args
    args = parse_args()

    # define seed, remove warning from log
    tf.get_logger().setLevel(logging.ERROR)
    tf.random.set_seed(123)
    np.random.seed(321)

    # init arguments
    model_name = "mlp" if args.name is None else args.name[0]
    train_set = args.train[0][:-4] if \
        len(args.train[0]) > 4 and args.train[0][-4:] == ".npy" else args.train[0]
    test_set = args.test[0][:-4] if \
        len(args.train[0]) > 4 and args.test[0][-4:] == ".npy" else args.test[0]

    # train loop over various configurations
    for name, monitor in [(f"{model_name}_v0", None),
                          (f"{model_name}_v1", "val_precision"),
                          (f"{model_name}_v2", "val_recall"),
                          (f"{model_name}_v3", "val_acc")]:
        metrics = [tf.keras.metrics.BinaryAccuracy(name='acc'),
                   tf.keras.metrics.Precision(name='precision'),
                   tf.keras.metrics.Recall(name="recall")]
        train_loop(args.w_dir[0], f"{train_set}.npy", f"{test_set}.npy", f"{name}_o1",
                   metrics, MLP, monitor=monitor, num_outputs=1)
        metrics = [tf.keras.metrics.CategoricalAccuracy(name='acc'),
                   tf.keras.metrics.Precision(name='precision'),
                   tf.keras.metrics.Recall(name="recall")]
        train_loop(args.w_dir[0], f"{train_set}_v2.npy", f"{test_set}.npy", f"{name}_o2",
                   metrics, MLP, monitor=monitor, num_outputs=2)
