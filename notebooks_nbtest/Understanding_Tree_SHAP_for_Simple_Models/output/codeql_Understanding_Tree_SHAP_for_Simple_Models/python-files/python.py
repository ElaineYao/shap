import nbtest
import json
import numpy as np
random_seed = np.random.randint(10000)
import graphviz
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_graphviz
import shap
N = 100
M = 4
X = np.zeros((N, M))
X.shape
y = np.zeros(N)
X[: N // 2, 0] = 1
y[: N // 2] = 1
single_split_model = DecisionTreeRegressor(max_depth=1)
single_split_model.fit(X, y)
dot_data = export_graphviz(
    single_split_model,
    out_file=None,
    filled=True,
    rounded=True,
    special_characters=True,
)
graph = graphviz.Source(dot_data)
graph
xs = [np.ones(M), np.zeros(M)]
df = pd.DataFrame()
for idx, x in enumerate(xs):
    index = pd.MultiIndex.from_product([[f"Example {idx}"], ["x", "shap_values"]])
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [x, shap.TreeExplainer(single_split_model).shap_values(x)],
                index=index,
                columns=["x0", "x1", "x2", "x3"],
            ),
        ]
    )
df
N = 100
M = 4
X = np.zeros((N, M))
X.shape
y = np.zeros(N)
X[: 1 * N // 4, 1] = 1
X[: N // 2, 0] = 1
X[N // 2 : 3 * N // 4, 1] = 1
y[: 1 * N // 4] = 1
and_model = DecisionTreeRegressor(max_depth=2)
and_model.fit(X, y)
dot_data = export_graphviz(and_model, out_file=None, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph
xs = np.array([np.ones(M), np.zeros(M)])
df = pd.DataFrame()
for idx, x in enumerate(xs):
    index = pd.MultiIndex.from_product([[f"Example {idx}"], ["x", "shap_values"]])
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [x, shap.TreeExplainer(and_model).shap_values(x)],
                index=index,
                columns=["x0", "x1", "x2", "x3"],
            ),
        ]
    )
df
y.mean()
N = 100
M = 4
X = np.zeros((N, M))
X.shape
y = np.zeros(N)
X[: N // 2, 0] = 1
X[: 1 * N // 4, 1] = 1
X[N // 2 : 3 * N // 4, 1] = 1
y[: N // 2] = 1
y[N // 2 : 3 * N // 4] = 1
or_model = DecisionTreeRegressor(max_depth=2)
or_model.fit(X, y)
dot_data = export_graphviz(or_model, out_file=None, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph
xs = np.array([np.ones(M), np.zeros(M)])
df = pd.DataFrame()
for idx, x in enumerate(xs):
    index = pd.MultiIndex.from_product([[f"Example {idx}"], ["x", "shap_values"]])
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [x, shap.TreeExplainer(or_model).shap_values(x)],
                index=index,
                columns=["x0", "x1", "x2", "x3"],
            ),
        ]
    )
df
N = 100
M = 4
X = np.zeros((N, M))
X.shape
y = np.zeros(N)
X[: N // 2, 0] = 1
X[: 1 * N // 4, 1] = 1
X[N // 2 : 3 * N // 4, 1] = 1
y[1 * N // 4 : N // 2] = 1
y[N // 2 : 3 * N // 4] = 1
xor_model = DecisionTreeRegressor(max_depth=2)
xor_model.fit(X, y)
dot_data = export_graphviz(xor_model, out_file=None, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph
xs = np.array([np.ones(M), np.zeros(M)])
df = pd.DataFrame()
for idx, x in enumerate(xs):
    index = pd.MultiIndex.from_product([[f"Example {idx}"], ["x", "shap_values"]])
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [x, shap.TreeExplainer(xor_model).shap_values(x)],
                index=index,
                columns=["x0", "x1", "x2", "x3"],
            ),
        ]
    )
df
N = 100
M = 4
X = np.zeros((N, M))
X.shape
y = np.zeros(N)
X[: N // 2, 0] = 1
X[: 1 * N // 4, 1] = 1
X[N // 2 : 3 * N // 4, 1] = 1
y[: 1 * N // 4] = 1
y[: N // 2] += 1
and_fb_model = DecisionTreeRegressor(max_depth=2)
and_fb_model.fit(X, y)
dot_data = export_graphviz(and_fb_model, out_file=None, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph
xs = np.array([np.ones(M), np.zeros(M)])
df = pd.DataFrame()
for idx, x in enumerate(xs):
    index = pd.MultiIndex.from_product([[f"Example {idx}"], ["x", "shap_values"]])
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [x, shap.TreeExplainer(and_fb_model).shap_values(x)],
                index=index,
                columns=["x0", "x1", "x2", "x3"],
            ),
        ]
    )
df
