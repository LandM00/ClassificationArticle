import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from collections import defaultdict
import matplotlib.pyplot as plt
from tensorflow.keras import mixed_precision

# Precisione mista per prestazioni su GPU
mixed_precision.set_global_policy("mixed_float16")

# Costanti
IMG_WIDTH, IMG_HEIGHT = 256, 192
BATCH_SIZE = 16
EPOCHS = 100
K_FOLDS = 5

# Classi e trattamenti
SPECIES = ["Aubergine", "Basil", "Cucumber", "Tomato"]
STAGES = ["GerminationStage", "VegetativeStage"]
LIGHT_TREATMENTS = ["RB1", "RB3", "RB5", "RB7", "RB9"]
DATASET_PATH = r"C:\Users\matte\OneDrive\Desktop\LavoriUNIBO\Lavori Rescue\MyArticle\Classification_Article\Dataset_light"

# Caricamento dataset
def collect_image_paths(dataset_path):
    paths, species_labels, stage_labels, light_labels = [], [], [], []
    for s_idx, species in enumerate(SPECIES):
        for st_idx, stage in enumerate(STAGES):
            for light in LIGHT_TREATMENTS:
                dir_path = os.path.join(dataset_path, species, stage, light)
                for fname in os.listdir(dir_path):
                    paths.append(os.path.join(dir_path, fname))
                    species_labels.append(s_idx)
                    stage_labels.append(st_idx)
                    light_labels.append(light)
    return np.array(paths), np.array(species_labels), np.array(stage_labels), np.array(light_labels)

# Dataset con preprocessing e augmentazione
def make_dataset(image_paths, species_labels, stage_labels, is_training=True):
    def process(path, s_lbl, st_lbl):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        img = tf.cast(img, tf.float32) / 255.0
        if is_training:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.05)
        return img, {"species_output": s_lbl, "stage_output": st_lbl}

    ds = tf.data.Dataset.from_tensor_slices((image_paths, species_labels, stage_labels))
    ds = ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        ds = ds.shuffle(1000)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Modello CNN migliorato
def create_cnn_model():
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # Blocchi convoluzionali
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout2D(0.3)(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Dense
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)

    # Output heads
    out_species = layers.Dense(len(SPECIES), activation='softmax', name='species_output')(x)
    out_stage = layers.Dense(len(STAGES), activation='softmax', name='stage_output')(x)

    model = keras.Model(inputs=inputs, outputs=[out_species, out_stage])

    # Ottimizzatore e compilazione
    base_opt = keras.optimizers.Adam(1e-3)
    optimizer = mixed_precision.LossScaleOptimizer(base_opt)
    model.compile(
        optimizer=optimizer,
        loss={'species_output': 'sparse_categorical_crossentropy', 'stage_output': 'sparse_categorical_crossentropy'},
        metrics={'species_output': 'accuracy', 'stage_output': 'accuracy'}
    )
    return model

# === MAIN ===

# Caricamento dati
image_paths, labels_species, labels_stages, light_labels = collect_image_paths(DATASET_PATH)
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

# Contenitori per metriche e confusion matrix
treatment_metrics = defaultdict(list)
conf_matrices_species = defaultdict(list)
conf_matrices_stage = defaultdict(list)

# K-Fold training
for fold, (train_idx, val_idx) in enumerate(kf.split(image_paths)):
    print(f"\n🔁 Fold {fold + 1}/{K_FOLDS}")

    train_ds = make_dataset(image_paths[train_idx], labels_species[train_idx], labels_stages[train_idx], True)
    val_ds = make_dataset(image_paths[val_idx], labels_species[val_idx], labels_stages[val_idx], False)

    model = create_cnn_model()
    callbacks = [
        keras.callbacks.ModelCheckpoint(f"model_fold_{fold+1}.h5", save_best_only=True, monitor='val_loss'),
        keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=1)

    # Previsione su validation set
    val_imgs, val_s_labels, val_st_labels, val_lights = [], [], [], []
    for idx in val_idx:
        img = tf.io.read_file(image_paths[idx])
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        img = tf.cast(img, tf.float32) / 255.0
        val_imgs.append(img.numpy())
        val_s_labels.append(labels_species[idx])
        val_st_labels.append(labels_stages[idx])
        val_lights.append(light_labels[idx])

    val_imgs = np.stack(val_imgs)
    pred_species, pred_stage = model.predict(val_imgs, verbose=0)
    pred_species = np.argmax(pred_species, axis=1)
    pred_stage = np.argmax(pred_stage, axis=1)

    # Metriche e confusion matrix per trattamento
    for treatment in LIGHT_TREATMENTS:
        idxs = [i for i, l in enumerate(val_lights) if l == treatment]
        if not idxs:
            continue

        y_true_s = np.array(val_s_labels)[idxs]
        y_true_st = np.array(val_st_labels)[idxs]
        y_pred_s = pred_species[idxs]
        y_pred_st = pred_stage[idxs]

        acc_s = accuracy_score(y_true_s, y_pred_s)
        acc_st = accuracy_score(y_true_st, y_pred_st)
        f1_s = f1_score(y_true_s, y_pred_s, average='macro')
        f1_st = f1_score(y_true_st, y_pred_st, average='macro')
        mcc_s = matthews_corrcoef(y_true_s, y_pred_s)
        mcc_st = matthews_corrcoef(y_true_st, y_pred_st)
        precision_s = precision_score(y_true_s, y_pred_s, average='macro', zero_division=0)
        precision_st = precision_score(y_true_st, y_pred_st, average='macro', zero_division=0)
        recall_s = recall_score(y_true_s, y_pred_s, average='macro', zero_division=0)
        recall_st = recall_score(y_true_st, y_pred_st, average='macro', zero_division=0)

        treatment_metrics[treatment].append(
            (acc_s, acc_st, f1_s, f1_st, mcc_s, mcc_st, precision_s, precision_st, recall_s, recall_st)
        )

        cm_s = confusion_matrix(y_true_s, y_pred_s, labels=list(range(len(SPECIES))))
        cm_st = confusion_matrix(y_true_st, y_pred_st, labels=list(range(len(STAGES))))
        conf_matrices_species[treatment].append(cm_s)
        conf_matrices_stage[treatment].append(cm_st)

# Report metriche aggregate
print("\n📊 Performance per trattamento luminoso (media sui fold):")
for treatment in LIGHT_TREATMENTS:
    acc_s, acc_st, f1_s, f1_st, mcc_s, mcc_st, precision_s, precision_st, recall_s, recall_st = map(np.mean, zip(*treatment_metrics[treatment]))
    print(f"{treatment}: Species Acc={acc_s:.3f}, Stage Acc={acc_st:.3f}, "
          f"Species F1={f1_s:.3f}, Stage F1={f1_st:.3f}, "
          f"Species MCC={mcc_s:.3f}, Stage MCC={mcc_st:.3f}")

# Plot confusion matrix finale per trattamento
print("\n📉 Confusion Matrix finale normalizzata per trattamento luminoso:")
for treatment in LIGHT_TREATMENTS:
    total_cm_s = sum(conf_matrices_species[treatment])
    total_cm_st = sum(conf_matrices_stage[treatment])

    norm_cm_s = total_cm_s.astype(float) / total_cm_s.sum(axis=1, keepdims=True)
    norm_cm_st = total_cm_st.astype(float) / total_cm_st.sum(axis=1, keepdims=True)

    disp_s = ConfusionMatrixDisplay(norm_cm_s, display_labels=SPECIES)
    disp_s.plot(cmap="Blues", xticks_rotation=45, values_format=".2f")
    plt.title(f"{treatment} - Species (Average, {K_FOLDS} fold)")
    plt.tight_layout()
    plt.savefig(f"final_conf_matrix_species_{treatment}.png")
    plt.close()

    disp_st = ConfusionMatrixDisplay(norm_cm_st, display_labels=STAGES)
    disp_st.plot(cmap="Greens", xticks_rotation=45, values_format=".2f")
    plt.title(f"{treatment} - Stage (Average, {K_FOLDS} fold)")
    plt.tight_layout()
    plt.savefig(f"final_conf_matrix_stage_{treatment}.png")
    plt.close()
