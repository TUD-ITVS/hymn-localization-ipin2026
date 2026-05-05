import os
from datetime import datetime

import pandas as pd
import tensorflow as tf

from hymn.config import RESNET, TRAINED_DIR
from hymn.methods.resnet.architectures import ResNetRegression
from hymn.methods.resnet.checkpoint import ModelCheckpoint
from hymn.plotting.diagnostics import plot_training_history


def model_training(X_train, X_test, y_train, y_test, shape, params):
    tf.keras.utils.set_random_seed(RESNET["random_seed"])

    model = ResNetRegression(input_shape=shape, dropout_rate=RESNET["dropout_rate"])
    model.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=RESNET["learning_rate"], clipnorm=1.0),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=9, restore_best_weights=True, verbose=1
    )
    save_checkpoint = ModelCheckpoint(save_dir=TRAINED_DIR)

    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=RESNET["epochs"],
        batch_size=RESNET["batch_size"],
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, save_checkpoint],
    )

    hist_df = pd.DataFrame(history.history)
    plot_training_history(hist_df)

    params_str = "-".join(map(str, params))
    ts = int(datetime.timestamp(datetime.now()))
    hist_json_file = os.path.join(
        TRAINED_DIR,
        f'history_Adamax_Batch{RESNET["batch_size"]}_LR{RESNET["learning_rate"]}'
        f'_DR{RESNET["dropout_rate"]}_{params_str}_{ts}.json',
    )
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)

    val_loss = model.evaluate(X_test, y_test)
    print(f'Validation Loss: {val_loss}')

    return history, hist_json_file