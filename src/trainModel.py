import os
import time
from datetime import timedelta, datetime
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Bug workaround source: https://stackoverflow.com/questions/38073432/how-to-suppress-verbose-tensorflow-logging
import tensorflow as tf
from keras import optimizers

from models import SiameseModel
from losses import max_margin_triplet_loss, soft_margin_triplet_loss
from dataset import Dataset
from utils import format_timedelta
from testModel import test

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# --- Set global variables --- #
BATCH_SIZE = 16
MARGIN = 0.5
EPOCHS = 10
BASE_MODEL = 'vgg16'
NETVLAD = True
MODEL_NAME = "cvm-net"
LOAD_WEIGHTS = False
WEIGHTS_PATH = f"/tf/notebooks/saved_models/{MODEL_NAME}"
LOSS_TYPE = "hard-margin"
LOSSES_PATH = f"/tf/notebooks/logs/{MODEL_NAME}/"
LOSSES_FILE = str(datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))

SUBSET = False
if SUBSET:
    gnd_images_path = "/tf/CVUSA/gnd_test"
    sat_images_path = "/tf/CVUSA/sat_test"
else:
    gnd_images_path = "/tf/CVUSA/gnd_train"
    sat_images_path = "/tf/CVUSA/sat_train"


def print_progress(epoch, step, total_steps, total_time, total_loss):
    if step == 0:
        return  # This stops a divide by zero error - remove at your peril!

    step_f = float(step)
    avg_time_step = total_time / step_f
    progress = int(100 * round((step_f / float(total_steps)), 2) / 2)
    eta = timedelta(seconds=np.multiply(avg_time_step, (total_steps - step)))
    print(f"\repoch:{epoch}  {step}/{total_steps} "
          f"[{progress * '='}>{(50 - progress) * ' '}] "
          f"loss: {np.round(total_loss / step_f, decimals=2)}    "
          f"{int(avg_time_step * 1000)}ms/step    "
          f"ETA: {format_timedelta(eta)}      ", end="")


def save_losses(losses: list) -> list:
    """
    :param losses: to save to location at LOSSES_PATH
    :return: a new empty list
    """
    print(f"saving losses to: {LOSSES_PATH}")
    if not os.path.exists(LOSSES_PATH):
        os.mkdir(LOSSES_PATH)

    path = os.path.join(LOSSES_PATH, LOSSES_FILE)
    with open(path, "a") as file:
        for loss in losses:
            file.write(loss + ",\n")

    return []


def save_weights(model: SiameseModel) -> None:
    print(f"\nsaving weights to: {WEIGHTS_PATH}")
    if not os.path.exists(WEIGHTS_PATH):
        os.mkdir(WEIGHTS_PATH)
    model.siamese_network.save(WEIGHTS_PATH)


def train(load_from_file: bool = False):

    dataset = Dataset(gnd_images_path=gnd_images_path,
                      sat_images_path=sat_images_path,
                      base_network=BASE_MODEL,
                      batch_size=BATCH_SIZE)
    train_data = dataset.load()

    model = SiameseModel(base_network=BASE_MODEL, netvlad=NETVLAD)

    if load_from_file:
        print("Loading weights from file")
        model.load(WEIGHTS_PATH)

    optimiser = optimizers.Adam(0.001)
    model.compile(optimizer=optimiser, weighted_metrics=[])

    for epoch in range(EPOCHS):

        total_steps = train_data.__len__()
        total_loss = -1
        losses = []
        start_time = time.time()

        for step, (gnd, sat_p, sat_n) in enumerate(train_data.as_numpy_iterator()):

            if gnd.shape[0] != BATCH_SIZE:
                print("\nOdd batch size found, skipping ...")
                continue

            # Mine hard triplets, rearranges the negatives to be hard
            sat_n = model.mine_hard_triplets(gnd, sat_p, sat_n)

            with tf.GradientTape() as tape:
                # Forward pass on the Hard Triplets
                ap_distance, an_distance = model.siamese_network((gnd, sat_p, sat_n))

                # Compute the loss
                if LOSS_TYPE == "hard-margin":
                    loss = max_margin_triplet_loss(ap_distance, an_distance, alpha=MARGIN)
                elif LOSS_TYPE == "logistic":
                    loss = soft_margin_triplet_loss(ap_distance, an_distance)
                else:
                    raise RuntimeError("Either 'hard-margin' or 'logistic' loss must be selected")

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

        print(f"completed epoch in {format_timedelta(timedelta(seconds=(time.time() - start_time)))}")

        # Save weights and losses each epoch
        save_weights(model)
        losses = save_losses(losses)

        test(model=model, model_name=MODEL_NAME)


if __name__ == "__main__":
    train(load_from_file=LOAD_WEIGHTS)
