import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import clear_output

class LivePlot(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

        plt.ion()  # interactive mode
        self.fig, self.axes = plt.subplots(1, 2, figsize=(12,5))

    def on_epoch_end(self, epoch, logs=None):
        self.train_loss.append(logs['loss'])
        self.val_loss.append(logs['val_loss'])
        self.train_acc.append(logs['accuracy'])
        self.val_acc.append(logs['val_accuracy'])

        clear_output(wait=True)

        for ax in self.axes:
            ax.cla()

        # Loss plot
        self.axes[0].plot(self.train_loss, 'o-', label='Training loss')
        self.axes[0].plot(self.val_loss, 'o-', label='Validation loss')
        self.axes[0].set_title('Loss over Epochs')
        self.axes[0].set_xlabel('Epoch')
        self.axes[0].set_ylabel('Loss')
        self.axes[0].legend()
        self.axes[0].grid(True, alpha=0.3)

        # Accuracy plot
        self.axes[1].plot(self.train_acc, 'o-', label='Training accuracy')
        self.axes[1].plot(self.val_acc, 'o-', label='Validation accuracy')
        self.axes[1].set_title('Accuracy over Epochs')
        self.axes[1].set_xlabel('Epoch')
        self.axes[1].set_ylabel('Accuracy')
        self.axes[1].legend()
        self.axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)