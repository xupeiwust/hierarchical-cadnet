import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
from src.network_single_feat import HierarchicalGCNN as HierGCNN
from src.helper import dataloader_single_feat as dataloader
import time


def test_step(x, y):
    test_logits = model(x, training=False)
    loss_value = loss_fn(y, test_logits)

    y_true = np.argmax(y.numpy(), axis=1)
    y_pred = np.argmax(test_logits.numpy(), axis=1)

    test_loss_metric.update_state(loss_value)
    test_acc_metric.update_state(y, test_logits)
    test_precision_metric.update_state(y, test_logits)
    test_recall_metric.update_state(y, test_logits)
    test_iou_metric.update_state(y, test_logits)

    return y_true, y_pred


def analysis_report(y_true, y_pred):
    target_names = ["Chamfer", "Through hole", "Triangular passage", "Rectangular passage", "6-sides passage",
                    "Triangular through slot", "Rectangular through slot", "Circular through slot",
                    "Rectangular through step", "2-sides through step", "Slanted through step", "O-ring", "Blind hole",
                    "Triangular pocket", "Rectangular pocket", "6-sides pocket", "Circular end pocket",
                    "Rectangular blind slot", "Vertical circular end blind slot", "Horizontal circular end blind slot",
                    "Triangular blind step", "Circular blind step", "Rectangular blind step", "Round"]
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

    print(classification_report(y_true, y_pred, labels=labels, target_names=target_names, digits=9))


if __name__ == '__main__':
    num_classes = 24
    units = 512
    num_epochs = 100
    learning_rate = 1e-2
    dropout_rate = 0.3
    batch_size = 64
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

    model.load_weights("checkpoint/lvl_7_adj_single_units_512_date_2020-12-15.ckpt")
    test_dataloader = dataloader("data/Single_Feature_70_15_15/test_sparse.h5")

    y_true_total = []
    y_pred_total = []

    start_time = time.time()

    for x_batch_test, y_batch_test in test_dataloader:
        one_hot_y = tf.one_hot(y_batch_test, depth=num_classes)
        y_true, y_pred = test_step(x_batch_test, one_hot_y)
        y_true_total = np.append(y_true_total, y_true)
        y_pred_total = np.append(y_pred_total, y_pred)

    print("Time taken: %.2fs" % (time.time() - start_time))

    analysis_report(y_true_total, y_pred_total)
    test_loss = test_loss_metric.result()
    test_acc = test_acc_metric.result()
    test_precision = test_precision_metric.result()
    test_recall = test_recall_metric.result()
    test_iou = test_iou_metric.result()

    #tf.summary.scalar('test_loss', test_loss, step=optimizer.iterations)
    #tf.summary.scalar('test_acc', test_acc, step=optimizer.iterations)

    test_loss_metric.reset_states()
    test_acc_metric.reset_states()
    test_precision_metric.reset_states()
    test_recall_metric.reset_states()
    test_iou_metric.reset_states()

    print(f"Test loss={test_loss}, Test acc={test_acc}, Precision={test_precision}, Recall={test_recall},"
          f" MIoU={test_iou}")