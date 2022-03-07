"""Script for training loop of Hierarchical CADNet (Edge)."""

import tensorflow as tf
import datetime as dt
from src.network_edge import HierarchicalGCNN as HierGCNN
from src.helper import dataloader_edge as dataloader


def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
        grads = tape.gradient(loss_value, model.trainable_variables)

    grads = [tf.clip_by_norm(g, 1.0) for g in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss_metric.update_state(loss_value)
    train_acc_metric.update_state(y, logits)


def val_step(x, y):
    val_logits = model(x, training=False)
    loss_value = loss_fn(y, val_logits)

    val_loss_metric.update_state(loss_value)
    val_acc_metric.update_state(y, val_logits)


if __name__ == '__main__':
    import time

    # User defined parameters.
    num_classes = 25
    num_layers = 6
    units = 512
    num_epochs = 100
    learning_rate = 1e-2
    dropout_rate = 0.3
    train_set_path = "data/training_MFCAD++.h5"
    val_set_path = "data/val_MFCAD++.h5"

    save_name = f'edge_lvl_{num_layers}_units_{units}_epochs_{num_epochs}_date_{dt.datetime.now().strftime("%Y-%m-%d")}'

    model = HierGCNN(units=units, rate=dropout_rate, num_classes=num_classes, num_layers=num_layers)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    summary_writer = tf.summary.create_file_writer(f'./log/{save_name}')

    train_loss_metric = tf.keras.metrics.Mean()
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    val_loss_metric = tf.keras.metrics.Mean()
    val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

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

        train_dataloader = dataloader(train_set_path)
        val_dataloader = dataloader(val_set_path)

        # Run training loop
        with summary_writer.as_default():
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataloader):
                one_hot_y = tf.one_hot(y_batch_train, depth=num_classes)
                train_step(x_batch_train, one_hot_y)

                # Log every 20 batches.
                if step % 20 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(train_loss_metric.result()))
                    )

            train_loss = train_loss_metric.result()
            train_acc = train_acc_metric.result()

            tf.summary.scalar('train_loss', train_loss, step=optimizer.iterations)
            tf.summary.scalar('train_acc', train_acc, step=optimizer.iterations)

            train_loss_metric.reset_states()
            train_acc_metric.reset_states()

            print(f"Train loss={train_loss}, Train acc={train_acc}")

            # Run validation loop
            for x_batch_val, y_batch_val in val_dataloader:
                one_hot_y = tf.one_hot(y_batch_val, depth=num_classes)
                val_step(x_batch_val, one_hot_y)

            val_loss = val_loss_metric.result()
            val_acc = val_acc_metric.result()

            # Save model if it has a better validation accuracy.
            if val_acc > max_val_acc:
                min_val_loss = float(val_loss)
                min_train_loss = float(train_loss)
                max_train_acc = float(train_acc)
                max_val_acc = float(val_acc)
                model.save_weights(f"checkpoint/{save_name}.ckpt")
                max_epoch = epoch

            tf.summary.scalar('val_loss', val_loss, step=optimizer.iterations)
            tf.summary.scalar('val_acc', val_acc, step=optimizer.iterations)

            val_loss_metric.reset_states()
            val_acc_metric.reset_states()

            print(f"Val loss={val_loss}, Val acc={val_acc}")
            print("Time taken: %.2fs" % (time.time() - start_time))

    print(f"Epoch={max_epoch+1}, Max train acc={max_train_acc}, Max val acc={max_val_acc}")
    print(f"Train loss={min_train_loss}, Val loss={min_val_loss}")

