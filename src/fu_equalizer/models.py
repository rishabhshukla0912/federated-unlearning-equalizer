"""Equalizer architectures. The paper uses build_exp2 (MMSE classifier)."""
import re
import tensorflow as tf
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.optimizers import Adam

from .config import WINDOW_SIZE, NUM_FEATURES, NUM_CLASSES, CENTER

LOCAL_LR = 1e-3
FIX_POOLING_HEAD = True   # center-crop head; False gives the GAP ablation


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.\\/>-]", "_", s)


def _pool_or_crop(x, name):
    if FIX_POOLING_HEAD:
        x = layers.Cropping1D(cropping=(CENTER, CENTER), name=f"{name}_crop")(x)
        return layers.Flatten(name=f"{name}_flat")(x)
    return layers.GlobalAveragePooling1D(name=f"{name}_gap")(x)


def build_exp6(name="exp6"):
    inp = Input(shape=(WINDOW_SIZE, NUM_FEATURES), name="complex_equalizer_input")
    x = layers.Conv1D(64, 3, padding="same", activation="relu", name="c1")(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu", name="c2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu", name="c3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    skip = x
    r = layers.Conv1D(128, 3, padding="same", activation="relu", name="rc1")(x)
    r = layers.BatchNormalization(name="rbn1")(r)
    r = layers.Conv1D(128, 3, padding="same", name="rc2")(r)
    r = layers.BatchNormalization(name="rbn2")(r)
    x = layers.Activation("relu", name="res_act")(layers.Add(name="res_add")([skip, r]))
    att = layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1,
                                    name="local_attention")(x, x)
    x = layers.LayerNormalization(name="att_norm")(layers.Add(name="att_add")([x, att]))
    x = layers.Cropping1D(cropping=(CENTER, CENTER), name="center_crop")(x)
    x = layers.Flatten(name="center_flat")(x)
    x = layers.Dense(256, activation="relu", name="d1")(x)
    x = layers.BatchNormalization(name="dbn")(x)
    x = layers.Dropout(0.15, name="drop")(x)
    x = layers.Dense(128, activation="relu", name="d2")(x)
    out = layers.Dense(2, activation="linear", name="complex_residual_output")(x)
    m = Model(inp, out, name=safe_name(name))
    m.compile(optimizer=Adam(LOCAL_LR), loss="mse",
              metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")])
    return m


def build_exp2(name="exp2"):
    inp = Input(shape=(WINDOW_SIZE, NUM_FEATURES), name="received_csi_mmse_input")
    x = layers.Conv1D(64, 3, padding="same", activation="relu", name="c1")(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu", name="c2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu", name="c3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Dropout(0.20, name="drop1")(x)
    x = _pool_or_crop(x, "head")
    x = layers.Dense(256, activation="relu", name="d1")(x)
    x = layers.Dropout(0.20, name="drop2")(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="qam_prediction")(x)
    m = Model(inp, out, name=safe_name(name))
    m.compile(optimizer=Adam(LOCAL_LR), loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def build_exp4(name="exp4"):
    inp = Input(shape=(WINDOW_SIZE, NUM_FEATURES), name="mmse_residual_input")
    x = layers.Conv1D(64, 3, padding="same", activation="relu", name="c1")(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu", name="c2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu", name="c3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    skip = x
    r = layers.Conv1D(128, 3, padding="same", activation="relu", name="rc1")(x)
    r = layers.Conv1D(128, 3, padding="same", name="rc2")(r)
    x = layers.Activation("relu", name="res_act")(layers.Add(name="res_add")([skip, r]))
    x = _pool_or_crop(x, "head")
    x = layers.Dense(128, activation="relu", name="d1")(x)
    x = layers.Dropout(0.15, name="drop")(x)
    out = layers.Dense(2, activation="linear", name="residual_correction")(x)
    m = Model(inp, out, name=safe_name(name))
    m.compile(optimizer=Adam(LOCAL_LR), loss="mse",
              metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")])
    return m


ALGORITHMS = {
    "exp6_complex_residual": dict(label="Exp 6 - complex residual", kind="regression",
                                  build=build_exp6, published=52.9695),
    "exp2_mmse_classifier":  dict(label="Exp 2 - MMSE classifier",  kind="classification",
                                  build=build_exp2, published=52.4301),
    "exp4_residual_cnn":     dict(label="Exp 4 - residual CNN",     kind="regression",
                                  build=build_exp4, published=52.6969),
}
