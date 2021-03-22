import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
from src.network_edge_residual import HierarchicalGCNN as HierGCNN
from src.helper import load_graphs_from_csv as dataloader
import time
import csv
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import os


def test_step(x, y, num_classes):
    test_logits = model(x, training=False)
    loss_value = loss_fn(y, test_logits)

    one_hot_logits = tf.one_hot(tf.math.argmax(test_logits, axis=1), depth=num_classes)

    iou = mean_iou(one_hot_logits, y)

    y_true = np.argmax(y.numpy(), axis=1)
    y_pred = np.argmax(test_logits.numpy(), axis=1)

    test_loss_metric.update_state(loss_value)
    test_acc_metric.update_state(y, test_logits)
    test_precision_metric.update_state(y, test_logits)
    test_recall_metric.update_state(y, test_logits)
    test_iou_metric.update_state(y, test_logits)

    return y_true, y_pred, iou


def analysis_report(y_true, y_pred):
    target_names = ["Chamfer", "Through hole", "Triangular passage", "Rectangular passage", "6-sides passage",
                    "Triangular through slot", "Rectangular through slot", "Circular through slot",
                    "Rectangular through step", "2-sides through step", "Slanted through step", "O-ring", "Blind hole",
                    "Triangular pocket", "Rectangular pocket", "6-sides pocket", "Circular end pocket",
                    "Rectangular blind slot", "Vertical circular end blind slot", "Horizontal circular end blind slot",
                    "Triangular blind step", "Circular blind step", "Rectangular blind step", "Round", "Stock"]
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

    print(classification_report(y_true, y_pred, labels=labels, target_names=target_names, digits=4))


def analysis_report_planar(y_true, y_pred):
    target_names = ["Rectangular through slot", "Triangular through slot", "Rectangular passage", "Triangular passage",
                    "6-sides passage", "Rectangular through step", "2-sides through step", "Slanted through step",
                    "Rectangular blind step", "Triangular blind step", "Rectangular blind slot", "Rectangular pocket",
                    "Triangular pocket", "6-sides pocket", "Chamfer", "Stock"]
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    print(classification_report(y_true, y_pred, labels=labels, target_names=target_names, digits=4))


def write_confusion_matrix(y_true, y_pred, num_classes):
    confusion_matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes).numpy()

    with open('confusion_matrix.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, quoting=csv.QUOTE_ALL)

        for i in range(np.shape(confusion_matrix)[0]):
            csvwriter.writerow(confusion_matrix[i])


def plot_confusion_matrix(y_true, y_pred, num_classes):
    if num_classes == 25:
        target_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17",
                        "18", "19", "20", "21", "22", "23", "24"]
    else:
        target_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]

    confusion_matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes).numpy()

    confusion_matrix = (confusion_matrix.T / confusion_matrix.sum(axis=1)).T * 100
    df_cm = pd.DataFrame(confusion_matrix, index=target_names,
                         columns=target_names)

    heatmap = sn.heatmap(df_cm, cmap=sn.cm._cmap_r, cbar_kws={'format': '%.0f%%'})
    heatmap.set_xlabel("Predicted Class", fontsize=13)
    heatmap.set_ylabel("True Class", fontsize=13)
    plt.show()


def intersection_and_union(pred, target):
    pred = tf.cast(pred, dtype=tf.bool)
    target = tf.cast(target, dtype=tf.bool)
    i = tf.reduce_sum(tf.cast(tf.math.logical_and(pred, target), tf.float32), axis=0)
    u = tf.reduce_sum(tf.cast(tf.math.logical_or(pred, target), tf.float32), axis=0)

    return i, u


def mean_iou(pred, target):
    i, u = intersection_and_union(pred, target)
    iou = i / u
    iou = iou.numpy()
    iou[np.isnan(iou)] = 1
    iou = np.mean(iou, axis=-1)

    return iou


def read_handle_csv(csv_path):
    handles = []

    with open(csv_path, 'r') as file:
        for i, line in enumerate(file):
            if i == 0:  # Skip first line (header)
                continue
            s = line[:-1].split(',')
            handles.append(s[1])

    return handles


def save_prediction(handle_csv_path, pred):
    handles = read_handle_csv(handle_csv_path)

    with open('prediction.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=",")
        csvwriter.writerow(["Handles", "Prediction"])

        for i in range(len(handles)):
            csvwriter.writerow([handles[i], pred[i]])


if __name__ == '__main__':
    num_classes = 25
    units = 512
    num_epochs = 100
    learning_rate = 1e-2
    dropout_rate = 0.3
    decay_rate = learning_rate / num_epochs
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate,
                                                                 decay_steps=100000, decay_rate=decay_rate)

    model = HierGCNN(units=units, rate=dropout_rate, num_classes=num_classes)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    test_loss_metric = tf.keras.metrics.Mean()
    test_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    test_precision_metric = tf.keras.metrics.Precision()
    test_recall_metric = tf.keras.metrics.Recall()
    test_iou_metric = tf.keras.metrics.MeanIoU(num_classes=num_classes)

    #model.load_weights("checkpoint/residual_lvl_7_adj_mixed_units_512_date_2021-01-11.ckpt")
    model.load_weights("checkpoint/residual_lvl_7_edge_mixed_units_512_date_2021-01-11.ckpt")

    test_dataloader = dataloader("data/Mixed_Complex/CSVs_to_run/")

    y_true_total = []
    y_pred_total = []
    ious = []

    start_time = time.time()

    for x_batch_test, y_batch_test in test_dataloader:

        one_hot_y = tf.one_hot(y_batch_test, depth=num_classes)
        y_true, y_pred, iou = test_step(x_batch_test, one_hot_y, num_classes)
        print(f"True: {y_true}")
        print(f"Pred: {y_pred}")
        save_prediction("data/Mixed_Complex/CSVs_to_run/0-0-0-0-0-4-8-14-20/0-0-0-0-0-4-8-14-20_taghandle.csv", y_pred)
        y_true_total = np.append(y_true_total, y_true)
        y_pred_total = np.append(y_pred_total, y_pred)

        ious += [iou]

    print("Time taken: %.2fs" % (time.time() - start_time))

    analysis_report(y_true_total, y_pred_total)
    test_loss = test_loss_metric.result()
    test_acc = test_acc_metric.result()
    test_precision = test_precision_metric.result()
    test_recall = test_recall_metric.result()
    test_iou = test_iou_metric.result()

    #write_confusion_matrix(y_true_total, y_pred_total, num_classes)
    #plot_confusion_matrix(y_true_total, y_pred_total, num_classes)

    #tf.summary.scalar('test_loss', test_loss, step=optimizer.iterations)
    #tf.summary.scalar('test_acc', test_acc, step=optimizer.iterations)

    test_loss_metric.reset_states()
    test_acc_metric.reset_states()
    test_precision_metric.reset_states()
    test_recall_metric.reset_states()
    test_iou_metric.reset_states()

    print(f"Test loss={test_loss}, Test acc={test_acc}, Precision={test_precision}, Recall={test_recall},"
          f" MIoU={test_iou}")

    print(f"Mean IoU: {np.mean(ious)}")
    print(ious)
