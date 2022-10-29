from sklearn.metrics import mean_squared_error, accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from pysr import PySRRegressor

df = pd.read_csv('star_classification.csv')

#mapping = {'GALAXY': 1, 'QSO': 2, 'STAR': 3}
df = df.replace({'GALAXY': 1, 'QSO': 2, 'STAR': 3})

y = df['class']
X = df.drop(['class', 'obj_ID'], axis=1)
model = PySRRegressor(
    model_selection="best",  # Result is mix of simplicity+accuracy
    niterations=50,
    populations=20,
    binary_operators=['plus', 'sub', 'mult', 'pow',
                      'div', 'mod'],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "inv(x) = 1/x",
        "square",
        "quart(x) = x^4",
        "cube",
        "abs",
        "log_abs",
        "log10_abs",
        "log2_abs",
        "log1p_abs",
        "sqrt_abs",
        "tan",
        "sinh",
        "cosh",
        "tanh",
        "atan",
        "asinh",
        "acosh_abs",
        "atanh_clip",
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / \
                          x, "quart": lambda x: x ** 4},
    loss="loss(x, y) = (x - y)^2",
    select_k_features=9,
    batch_size=9500,
    batching=True,
    early_stop_condition=(
        "stop_if(loss, complexity) = loss < 1e-2 && complexity < 30"
        # Stop early if we find a good and simple equation
    ),
    timeout_in_seconds=60 * 60 * 24,
    warm_start=True,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=54)
model.fit(X_train, y_train)
print(model)

pickle.dump(model, open('model.pkl', 'wb'))

y_pred = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
y_pred = [np.round(x) for x in y_pred]
print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")
