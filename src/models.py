"""
models.py
=========
Phase 3: supervised models that predict the `cluster_label` from Phase 2.

We train three models so we can compare them (the rubric requires at least two,
one of which must be a neural network):
    1. Logistic Regression  -> simple, interpretable baseline.
    2. Random Forest        -> strong classical model, little tuning needed.
    3. Neural Network (MLP) -> the mandatory deep-learning model (Keras).

Each builder returns a trained model. Evaluation/comparison lives in evaluation.py
so this file stays focused on *defining and training* the models.
"""

from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

from preprocessing import RANDOM_STATE


def train_logistic_regression(X_train, y_train):
    """Baseline classifier: multinomial logistic regression.

    It draws linear decision boundaries between classes. It's the 'floor': if a
    more complex model can't beat it, that complex model isn't adding value.
    `class_weight='balanced'` makes minority classes count more, countering the
    imbalance in the data.
    """
    model = LogisticRegression(
        multi_class="multinomial",
        class_weight="balanced",
        max_iter=1000,            # more iterations so it converges on scaled data
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """Robust classical model: an ensemble of decision trees.

    A Random Forest trains many trees on random subsets of data/features and
    averages their votes. It captures non-linear patterns, rarely overfits badly,
    and needs little tuning. Often the strongest model on tabular data like this.
    """
    model = RandomForestClassifier(
        n_estimators=300,         # number of trees
        max_depth=None,           # let trees grow fully
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,                # use all CPU cores
    )
    model.fit(X_train, y_train)
    return model


def build_neural_network(input_dim: int, n_classes: int):
    """Define (compile) the mandatory neural network with Keras.

    Architecture: two hidden layers with regularization to fight overfitting.
      - Dense(relu)        -> learns non-linear combinations of the features.
      - BatchNormalization -> stabilizes/accelerates training.
      - Dropout(0.3)       -> randomly drops 30% of neurons each step so the net
                              doesn't memorize the training data.
      - softmax output     -> outputs a probability per class.
    Imported here (not at top) so the heavy TensorFlow import only loads when
    a neural network is actually built.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",   # integer labels (0,1,2,...)
        metrics=["accuracy"],
    )
    return model


def train_neural_network(X_train, y_train, X_val, y_val,
                         input_dim: int, n_classes: int,
                         epochs: int = 100, batch_size: int = 32):
    """Train the neural network with early stopping on the validation set.

    This is exactly why we kept a separate validation split: EarlyStopping watches
    the validation loss and stops when it stops improving (restoring the best
    weights), and ReduceLROnPlateau lowers the learning rate when training stalls.
    Returns the trained model and the Keras `history` (loss/accuracy per epoch) so
    we can plot the learning curves in the report.
    """
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    model = build_neural_network(input_dim, n_classes)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return model, history


def save_sklearn_model(model, path: str):
    """Persist a scikit-learn model (Logistic Regression / Random Forest)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def save_keras_model(model, path: str = "models/nn_model.h5"):
    """Persist the Keras neural network (different format from joblib)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
