import tensorflow as tf
import numpy as np
import datetime as dt
from src.network_edge_residual import HierarchicalGCNN as HierGCNN
from src.helper import dataloader_edge as dataloader


def train_step_weighted(x, hot_y, y):
    sample_weights = np.take(class_weights, y)

    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(hot_y, logits, sample_weight=sample_weights)
        grads = tape.gradient(loss_value, model.trainable_variables)

    grads = [tf.clip_by_norm(g, 1.0) for g in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss_metric.update_state(loss_value)
    train_acc_metric.update_state(hot_y, logits)
    train_miou_metric.update_state(hot_y, logits)


def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
        grads = tape.gradient(loss_value, model.trainable_variables)

    grads = [tf.clip_by_norm(g, 1.0) for g in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss_metric.update_state(loss_value)
    train_acc_metric.update_state(y, logits)
    train_miou_metric.update_state(y, logits)


def val_step(x, y):
    val_logits = model(x, training=False)
    loss_value = loss_fn(y, val_logits)

    val_loss_metric.update_state(loss_value)
    val_acc_metric.update_state(y, val_logits)
    val_miou_metric.update_state(y, val_logits)


if __name__ == '__main__':
    import time

    data_type = "MFCAD++"
    if data_type == "MFCAD":
        num_classes = 16
    else:
        print("MFCAD++")
        num_classes = 25

    class_weights = np.array([0.049985, 0.049299, 0.016895, 0.012569, 0.00843, 0.073304, 0.045495, 0.134623,
                             0.026871, 0.018387, 0.026435, 0.01925, 0.028516, 0.014515, 0.011091, 0.008496,
                             0.011977, 0.053868, 0.067164, 0.07911, 0.02704, 0.027974, 0.01833, 0.167209, 0.003167])

    num_layers = 7
    units = 512
    num_epochs = 100
    learning_rate = 1e-2
    dropout_rate = 0.3
    decay_rate = learning_rate / num_epochs
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate,
                                                                 decay_steps=100000, decay_rate=decay_rate)

    save_name = f'MF_CAD++_residual_lvl_{num_layers}_edge_{data_type}_units_{units}_date_{dt.datetime.now().strftime("%Y-%m-%d")}_epochs_{num_epochs}'

    model = HierGCNN(units=units, rate=dropout_rate, num_classes=num_classes, num_layers=num_layers)
    #model.load_weights("checkpoint/residual_planar_planar_units_512_date_2020-11-11.ckpt")

    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    summary_writer = tf.summary.create_file_writer(f'./log/{save_name}')

    train_loss_metric = tf.keras.metrics.Mean()
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    train_miou_metric = tf.keras.metrics.MeanIoU(num_classes=num_classes)
    val_loss_metric = tf.keras.metrics.Mean()
    val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    val_miou_metric = tf.keras.metrics.MeanIoU(num_classes=num_classes)

    min_val_loss = 0.0
    min_train_loss = 0.0
    max_train_acc = 0.0
    max_val_acc = 0.0
    max_epoch = 0
    max_train_miou = 0.0
    max_val_miou = 0.0

    for epoch in tf.range(num_epochs):
        print(f"Epoch {epoch + 1} of {num_epochs}")
        start_time = time.time()

        train_dataloader = dataloader("/home/mlg/Documents/Andrew/Datasets/MFCAD++/hierarchical_graphs/training_MFCAD++.h5")
        val_dataloader = dataloader("/home/mlg/Documents/Andrew/Datasets/MFCAD++/hierarchical_graphs/val_MFCAD++.h5")

        with summary_writer.as_default():
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataloader):
                one_hot_y = tf.one_hot(y_batch_train, depth=num_classes)
                #train_step_weighted(x_batch_train, one_hot_y, y_batch_train)
                train_step(x_batch_train, one_hot_y)
                #model.summary()

                # Log every 20 batches.
                if step % 20 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(train_loss_metric.result()))
                    )

            train_loss = train_loss_metric.result()
            train_acc = train_acc_metric.result()
            train_miou = train_miou_metric.result()

            tf.summary.scalar('train_loss', train_loss, step=optimizer.iterations)
            tf.summary.scalar('train_acc', train_acc, step=optimizer.iterations)
            tf.summary.scalar('train_miou', train_miou, step=optimizer.iterations)

            train_loss_metric.reset_states()
            train_acc_metric.reset_states()
            train_miou_metric.reset_states()

            print(f"Train loss={train_loss}, Train acc={train_acc}, Train MIoU={train_miou}")

            for x_batch_val, y_batch_val in val_dataloader:
                one_hot_y = tf.one_hot(y_batch_val, depth=num_classes)
                val_step(x_batch_val, one_hot_y)

            val_loss = val_loss_metric.result()
            val_acc = val_acc_metric.result()
            val_miou = val_miou_metric.result()

            if val_acc > max_val_acc:
                min_val_loss = float(val_loss)
                min_train_loss = float(train_loss)
                max_train_acc = float(train_acc)
                max_val_acc = float(val_acc)
                max_train_miou = float(train_miou)
                max_val_miou = float(val_miou)
                model.save_weights(f"checkpoint/{save_name}.ckpt")
                max_epoch = epoch

            tf.summary.scalar('val_loss', val_loss, step=optimizer.iterations)
            tf.summary.scalar('val_acc', val_acc, step=optimizer.iterations)
            tf.summary.scalar('val_miou', val_miou, step=optimizer.iterations)

            val_loss_metric.reset_states()
            val_acc_metric.reset_states()
            val_miou_metric.reset_states()

            print(f"Val loss={val_loss}, Val acc={val_acc}, Val MIoU={val_miou}")
            print("Time taken: %.2fs" % (time.time() - start_time))

    print(f"Epoch={max_epoch+1}, Max train acc={max_train_acc}, Max val acc={max_val_acc}")
    print(f"Train loss={min_train_loss}, Val loss={min_val_loss}")
    print(f"Max train MIoU={max_train_miou}, Max val MIoU={max_val_miou}")

