import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
from src.network_edge_residual import HierarchicalGCNN as HierGCNN
from src.helper import dataloader_edge as dataloader
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


def analysis_report_mfcad(y_true, y_pred):
    target_names = ["Chamfer", "Triangular passage", "Rectangular passage", "6-sides passage", "Triangular through slot",
                    "Rectangular through slot", "Rectangular through step", "2-sides through step", "Slanted through step",
                    "Triangular pocket", "Rectangular pocket", "6-sides pocket", "Rectangular blind slot",
                    "Triangular blind step", "Rectangular blind step", "Stock"]

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


def save_prediction(name, labels, pred):
    with open("/home/mlg/Documents/Andrew/hierarchical-cadnet/result_labels/" + name + '_pred.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        csv_writer.writerow(["Groundtruth", "Prediction"])

        for i in range(len(labels)):
            csv_writer.writerow([labels[i], pred[i]])


def face_upvoting(facet_to_face, face_labels_true, facet_preds):
    face_labels = {}
    face_labels_upvote = []

    m, n = facet_to_face.shape

    for i in range(m):
        face_index = np.nonzero(facet_to_face[i])[0][0]

        if face_index not in face_labels:
            face_labels[face_index] = [facet_preds[i]]
        else:
            face_labels[face_index].append(facet_preds[i])

    for key, value in face_labels.items():
        counts = np.bincount(np.array(value))
        face_labels_upvote.append(np.argmax(counts))

    batch_pred = np.array(face_labels_upvote)
    c_faces = np.sum((batch_pred == face_labels_true))
    t_faces = np.size(face_labels_true)

    return c_faces, t_faces


if __name__ == '__main__':
    num_classes = 25
    num_layers = 7
    units = 512
    num_epochs = 100
    learning_rate = 1e-2
    dropout_rate = 1.0
    decay_rate = learning_rate / num_epochs
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate,
                                                                 decay_steps=100000, decay_rate=decay_rate)

    model = HierGCNN(units=units, rate=dropout_rate, num_classes=num_classes, num_layers=num_layers)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    test_loss_metric = tf.keras.metrics.Mean()
    test_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    test_precision_metric = tf.keras.metrics.Precision()
    test_recall_metric = tf.keras.metrics.Recall()
    test_iou_metric = tf.keras.metrics.MeanIoU(num_classes=num_classes)

    model.load_weights("checkpoint/ablation/MF_CAD++_edge_lvl_7_edge_MFCAD++_units_512_date_2021-09-01_epochs_100.ckpt")
    #print(model.summary())

    test_dataloader = dataloader("/home/mlg/Documents/Andrew/Datasets/MFCAD++/hierarchical_graphs/test_MFCAD++.h5")

    y_true_total = []
    y_pred_total = []
    ious = []
    correct_faces = total_faces = 0

    start_time = time.time()

    for x_batch_test, y_batch_test, CAD_models, idxs in test_dataloader:
        one_hot_y = tf.one_hot(y_batch_test, depth=num_classes)
        y_true, y_pred, iou = test_step(x_batch_test, one_hot_y, num_classes)

        previous_idx = 0

        for i in range(len(CAD_models)):
            name = str(CAD_models[i])[2:-1]
            model_idx = idxs[i][0] + 1

            slice_true = y_true[previous_idx:model_idx]
            slice_pred = y_pred[previous_idx:model_idx]
            previous_idx = model_idx
            save_prediction(name, slice_true, slice_pred)

        break

        #num_correct_faces, num_total_faces = face_upvoting(A_3, labels_face, y_pred)
        #correct_faces += num_correct_faces
        #total_faces += num_total_faces

        y_true_total = np.append(y_true_total, y_true)
        y_pred_total = np.append(y_pred_total, y_pred)

        ious += [iou]

    print("Time taken: %.2fs" % (time.time() - start_time))

    #print(f"Test acc per face: {correct_faces / total_faces}")
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
