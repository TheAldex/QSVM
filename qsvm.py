from qiskit import BasicAer
from sklearn.svm import SVC
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.circuit.library import ZZFeatureMap
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.datasets import ad_hoc_data
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings('ignore')

num_qubits = 2
seed = 10598

featured_dim = 2

X_train, y_train, X_test, y_test, class_labels = ad_hoc_data(
    training_size = 100,
    test_size = 50,
    n = featured_dim,
    gap = 0.3,
    plot_data = True,
    one_hot = False,
    include_sample_total = True,
)

plt.figure(figsize=(5, 5))
plt.ylim(0, 2 * np.pi)
plt.xlim(0, 2 * np.pi)
plt.imshow(
    np.asmatrix(class_labels).T,
    interpolation="nearest",
    origin="lower",
    cmap="RdBu",
    extent=[0, 2 * np.pi, 0, 2 * np.pi],
)

# A train plot
plt.scatter(
    X_train[
        np.where(y_train[:] == 0), 0
    ],  # x coordinate of y_train where class is 0
    X_train[
        np.where(y_train[:] == 0), 1
    ],  # y coordinate of y_train where class is 0
    marker="s",
    facecolors="w",
    edgecolors="b",
    label="A train",
)

# B train plot
plt.scatter(
    X_train[
        np.where(y_train[:] == 1), 0
    ],  # x coordinate of y_train where class is 1
    X_train[
        np.where(y_train[:] == 1), 1
    ],  # y coordinate of y_train where class is 1
    marker="o",
    facecolors="w",
    edgecolors="r",
    label="B train",
)

# A test plot
plt.scatter(
    X_test[np.where(y_test[:] == 0), 0],  # x coordinate of y_test where class is 0
    X_test[np.where(y_test[:] == 0), 1],  # y coordinate of y_test where class is 0
    marker="s",
    facecolors="b",
    edgecolors="w",
    label="A test",
)

# B test plot
plt.scatter(
    X_test[np.where(y_test[:] == 1), 0],  # x coordinate of y_test where class is 1
    X_test[np.where(y_test[:] == 1), 1],  # y coordinate of y_test where class is 1
    marker="o",
    facecolors="r",
    edgecolors="w",
    label="B test",
)

plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
plt.title("Ad hoc dataset for classification")

plt.show()

backend = QuantumInstance(BasicAer.get_backend("qasm_simulator"),
    shots=1024, seed_simulator=seed, seed_transpiler=seed
)
featured_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2, entanglement="linear")
kernel = QuantumKernel(feature_map=featured_map, quantum_instance=backend)

qsvc = QSVC(quantum_kernel=kernel)
qsvc.fit(X_train, y_train)
qsvc_score = qsvc.score(X_test, y_test)
qsvc_predict = qsvc.predict(X_test)

# kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’
svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)
svc_score = svc.score(X_test, y_test)
svc_predict = svc.predict(X_test)

print(classification_report(y_test, svc_predict))
print(f"Classical Kernel SVC classification test score: {svc_score}")

print("---------------------------------------------------------------")

print(classification_report(y_test, qsvc_predict))
print(f"Quantum Kernel QSVC classification test score: {qsvc_score}")