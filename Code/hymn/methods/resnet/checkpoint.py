import os
import tensorflow as tf

from hymn.config import TRAINED_DIR


class ModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, save_dir=TRAINED_DIR):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)  # Create directory if not exists
        self.best_val_loss = float('inf')  # Initialize best validation loss

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        # Get current losses
        val_loss = logs.get("val_loss", 0)

        # Save model only if val_loss has improved
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss  # Update best loss

            # Create a filename with epoch, train_loss, and val_loss
            filename = f"epoch_{epoch+1:02d}_valLoss-{val_loss:.4f}.h5"
            filepath = os.path.join(self.save_dir, filename)

            # Save the model
            self.model.save(filepath)
            print(f"\nImproved val_loss: {val_loss:.4f}. Model saved: {filepath}")
        else:
            print("\n val_loss did not improve. Model not saved.")