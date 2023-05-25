import os
import time
from datetime import timedelta, datetime
import numpy as np
import tensorflow as tf
from keras import optimizers
from models import SiameseModel
from utils import load_data, format_timedelta

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print("Tensorflow version:", tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))

# --- Set global variables --- #
BATCH_SIZE = 16
EPOCHS = 10
WEIGHTS_PATH = "/tf/notebooks/cvm-net"
LOSSES_PATH = "/tf/notebooks/logs/" + str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))


def print_progress(epoch, step, total_steps, total_time, total_loss):
    if step == 0:
        return  # This stops a divide by zero error - remove at your peril!

    step_f = float(step)
    avg_time_step = total_time / step_f
    progress = int(100 * round((step_f / float(total_steps)), 2) / 2)
    eta = timedelta(seconds=np.multiply(avg_time_step, (total_steps - step)))
    print(f"\repoch:{epoch}  {step}/{total_steps} "
          f"[{progress * '='}>{(50 - progress) * ' '}] "
          f"loss: {np.round(total_loss / step_f, decimals=2)}\t"
          f"{int(avg_time_step * 1000)}ms/step\t"
          f"ETA: {format_timedelta(eta)} ", end="")


def train(load_from_file: bool = False):
    train_data = load_data(anchor_images_path="/tf/CVUSA/terrestrial",  # "/tf/CVUSA/clean_ground",
                           positive_images_path="/tf/CVUSA/satellite",  # "/tf/CVUSA/clean_aerial",
                           batch_size=BATCH_SIZE)

    model = SiameseModel()

    if load_from_file:
        model.load(WEIGHTS_PATH)

    optimiser = optimizers.Adam(0.001)
    model.compile(optimizer=optimiser, weighted_metrics=[])

    for epoch in range(EPOCHS):

        total_steps = train_data.__len__()
        total_loss = -1
        losses = []
        start_time = time.time()

        for step, (gnd, sat_p, sat_n) in enumerate(train_data.as_numpy_iterator()):

            # Mine hard triplets, rearranges the negatives to be hard
            sat_n = model.mine_hard_triplets(gnd, sat_p, sat_n)

            with tf.GradientTape() as tape:
                # Forward pass on the Hard Triplets
                ap_distance, an_distance = model.siamese_network((gnd, sat_p, sat_n))

                # Compute the loss
                loss = ap_distance - an_distance
                loss = tf.maximum(loss + model.margin, 0.0)

                # Save the loss for updates/metrics
                losses.append(str(np.mean(loss)))
                if total_loss == -1:
                    total_loss = np.mean(loss)
                else:
                    total_loss += np.mean(loss)

            # Apply gradients to model
            grads = tape.gradient(loss, model.siamese_network.trainable_weights)
            optimiser.apply_gradients(zip(grads, model.siamese_network.trainable_weights))

            # Calculate the time per step
            total_time = np.subtract(time.time(), start_time)

            # Output progress update
            print_progress(epoch, step, total_steps, total_time, total_loss)

        # Save weights and losses each epoch
        print(f"\nsaving weights to: {WEIGHTS_PATH}")
        model.siamese_network.save(WEIGHTS_PATH)

        print(f"saving losses to: {LOSSES_PATH}")
        with open(LOSSES_PATH, "a") as file:
            for loss in losses:
                file.write(loss + ",\n")
            losses = []


if __name__ == "__main__":
    train(load_from_file=False)
