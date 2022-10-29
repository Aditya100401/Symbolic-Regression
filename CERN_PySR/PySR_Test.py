import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pysr import PySRRegressor

df = pd.read_csv('dielectron.csv')
df.dropna(inplace=True)


y = df['M']
X = df.drop(['M'], axis=1)
model = PySRRegressor(
    model_selection="best",  # Result is mix of simplicity+accuracy
    niterations=10,
    populations=20,
    binary_operators=['plus', 'sub', 'mult', 'pow',
                      'div', 'mod', 'logical_or', 'logical_and'],
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
    # select_k_features=9,
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
