from pathlib import Path
import numpy as np
import pandas as pd
import time
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mutual_info_score
from dwave_qbsolv import QBSolv

warnings.filterwarnings("ignore")

SEED         = 42
CV_FOLDS     = 10
K_TARGET     = 5       
B_BINS       = 20      
EPS          = 1e-8    
Lambda_high  = 50
Lambda_low   = -50
Lambda_iters = 25

DATA_VARIANT = "standard"

DATA_DIR  = Path(__file__).resolve().parent
DATA_PATH = {
    "standard": DATA_DIR / "waveform.data",
    "noise":    DATA_DIR / "waveform-+noise.data",
}[DATA_VARIANT]


class QBSolver:
    def __init__(self, Q_sym):
        self.Q_sym = Q_sym

    def solve(self, seed=42):
        n = self.Q_sym.shape[0]
        Q_dict = {(i, i): self.Q_sym[i, i] for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                Q_dict[(i, j)] = 2 * self.Q_sym[i, j]
        response = QBSolv().sample_qubo(Q_dict, seed=seed)
        return np.array([response.first.sample[i] for i in range(n)])


class MueckeQFS:
    def __init__(self, k_target=K_TARGET, B=B_BINS, epsilon=EPS,
                 max_iters=25, solver_cls=QBSolver):
        self.k       = k_target
        self.B       = B
        self.eps     = epsilon
        self.max_iters = max_iters
        self.solver_cls = solver_cls

    def fit_transform(self, X, y, seed=SEED):
        d = X.shape[1]

        X_binned = np.zeros_like(X, dtype=int)
        for j in range(d):
            qs    = np.linspace(0, 1, self.B + 1)
            edges = np.quantile(X[:, j], qs)
            if np.allclose(edges, edges[0]):
                X_binned[:, j] = 0
            else:
                X_binned[:, j] = np.digitize(X[:, j], edges[1:-1])

        I = np.array([mutual_info_score(X_binned[:, i], y) for i in range(d)])

        R = np.zeros((d, d))
        active = np.where(I > 0)[0]
        for idx1, i in enumerate(active):
            for j in active[idx1 + 1:]:
                val = mutual_info_score(X_binned[:, i], X_binned[:, j])
                R[i, j] = R[j, i] = val

        a_low, a_high = 0.0, 1.0
        best_x, best_diff = None, float('inf')

        print(f"\nMuecke QFS binary search (target k={self.k}):")
        for step in range(self.max_iters):
            alpha = (a_low + a_high) / 2.0

            Q = np.zeros((d, d))
            for i in range(d):
                for j in range(i + 1, d):
                    Q[i, j] = Q[j, i] = (1.0 - alpha) * R[i, j]

            mu = np.max(Q) if np.max(Q) > 0 else 1e-4

            for i in range(d):
                Q[i, i] = mu if alpha * I[i] < self.eps else -alpha * I[i]

            x     = self.solver_cls(Q).solve(seed=seed + step)
            k_sel = int(np.sum(x))
            print(f"  step {step:2d}: alpha={alpha:.6f}, k_sel={k_sel}")

            if abs(k_sel - self.k) < best_diff:
                best_diff = abs(k_sel - self.k)
                best_x = x.copy()

            if k_sel == self.k:
                break
            elif k_sel > self.k:
                a_high = alpha
            else:
                a_low = alpha

        selected = np.where(best_x == 1)[0]
        print(f"Muecke selected features: {selected}")
        return selected


class TaylorQFS:
    def __init__(self, k_target=K_TARGET, max_iters=Lambda_iters,
                 solver_cls=None):
        self.k_target   = k_target
        self.max_iters  = max_iters
        self.solver_cls = solver_cls

    def fit_transform(self, X, y, seed=SEED):
        d       = X.shape[1]
        classes = np.unique(y)
        K       = len(classes)

        normal_scaler = StandardScaler()
        X_scaled = normal_scaler.fit_transform(X)

        skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        Q_base  = np.zeros((d, d))
        n_folds = 0

        for train_idx, val_idx in skf.split(X_scaled, y):
            X_tr, y_tr   = X_scaled[train_idx], y[train_idx]
            X_val, y_val = X_scaled[val_idx],   y[val_idx]

            Q_fold = np.zeros((d, d))

            for cls in classes:
                y_binary = (y_tr == cls).astype(int)

                mdl_k = LogisticRegression(
                    C=0.1, solver="liblinear", max_iter=50).fit(X_tr, y_binary)

                w_k = mdl_k.coef_[0]                    
                p_k = mdl_k.predict_proba(X_val)[:, 1] 
                y_k = (y_val == cls).astype(float)

                g   = w_k * (X_val.T @ (y_k - p_k))
                W   = np.diag(p_k * (1.0 - p_k))
                H   = 0.5 * (X_val.T @ W @ X_val)


                Q_k = H.copy()
                np.fill_diagonal(Q_k, np.diag(H) - g)
                Q_fold += Q_k

            Q_fold  /= K        
            Q_base  += Q_fold   
            n_folds += 1

        Q_norm = Q_base / n_folds

        lam_low, lam_high = Lambda_low, Lambda_high
        best_x      = None
        best_x_over = None

        print(f"\nTaylorQFS binary search (target k={self.k_target}):")
        for step in range(self.max_iters):
            lam       = (lam_low + lam_high) / 2.0
            Q_current = Q_norm.copy()
            np.fill_diagonal(Q_current, np.diag(Q_norm) + lam)

            x     = self.solver_cls(Q_current).solve(seed=seed + step)
            k_sel = int(np.sum(x))
            print(f"  step {step:2d}: lambda={lam:8.3f}, k_sel={k_sel}")

            if k_sel >= self.k_target:
                if best_x_over is None or k_sel < int(np.sum(best_x_over)):
                    best_x_over = x.copy()

            if k_sel == self.k_target:
                best_x = x.copy()
                break
            elif k_sel > self.k_target:
                lam_low = lam
            else:
                lam_high = lam

        if best_x is None and best_x_over is not None:
            print(f"TaylorQFS: exact count {self.k_target} not reached. "
                  f"Pruning from {int(np.sum(best_x_over))}...")
            indices   = np.where(best_x_over == 1)[0]
            diag_vals = np.diag(Q_norm)[indices]
            top_k     = indices[np.argsort(diag_vals)[:self.k_target]]
            best_x    = np.zeros(d)
            best_x[top_k] = 1
        elif best_x is None:
            best_x = self.solver_cls(Q_norm).solve(seed=seed)

        selected = np.where(best_x == 1)[0]
        print(f"TaylorQFS selected features: {selected}")
        return selected

df   = pd.read_csv(DATA_PATH, header=None)
X_wv = df.iloc[:, :-1].values.astype(float)
y_wv = df.iloc[:, -1].values.astype(int)   

n_features = X_wv.shape[1]

print("=" * 60)
print(f"WAVEFORM DATASET  [{DATA_VARIANT.upper()}]")
print("=" * 60)
print(f"X shape : {X_wv.shape}  (n={X_wv.shape[0]} samples, d={n_features} features)")
print(f"Classes : {np.unique(y_wv)}  - balance: "
      + ", ".join(f"class {c}={np.mean(y_wv==c):.3f}" for c in np.unique(y_wv)))
print(f"Target k={K_TARGET} features")


FS_METHODS = {
    "Muecke_QBSolv":  MueckeQFS(k_target=K_TARGET, solver_cls=QBSolver),
    "TaylorS_QBSolv": TaylorQFS(k_target=K_TARGET, solver_cls=QBSolver)
}

selections = {}
timing     = {}
for name, algo in FS_METHODS.items():
    t0             = time.time()
    sel            = algo.fit_transform(X_wv, y_wv, SEED)
    timing[name]   = time.time() - t0
    selections[name] = sel


def make_classifiers(k):
    return {
        "LR": LogisticRegression(C=0.1, solver='liblinear', max_iter=500,multi_class='ovr', random_state=SEED),
    }


print("\n" + "=" * 60)
print("10-FOLD CV EVALUATION")
print("=" * 60)

cv      = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
clf_names = [ "LR"]

results = {fs: {clf: [] for clf in clf_names} for fs in FS_METHODS}

for fold, (tr_idx, te_idx) in enumerate(cv.split(X_wv, y_wv)):
    X_tr_raw, y_tr = X_wv[tr_idx], y_wv[tr_idx]
    X_te_raw, y_te = X_wv[te_idx], y_wv[te_idx]

    scaler = StandardScaler().fit(X_tr_raw)
    X_tr   = scaler.transform(X_tr_raw)
    X_te   = scaler.transform(X_te_raw)

    for fs_name, sel in selections.items():
        if len(sel) == 0:
            for clf_name in results[fs_name]:
                results[fs_name][clf_name].append(0.0)
            continue

        X_tr_sel = X_tr[:, sel]
        X_te_sel = X_te[:, sel]
        clfs     = make_classifiers(len(sel))

        for clf_name, clf in clfs.items():
            clf.fit(X_tr_sel, y_tr)
            preds = clf.predict(X_te_sel)
            results[fs_name][clf_name].append(accuracy_score(y_te, preds))


print("\n" + "=" * 60)
print(f"RESULTS  (10-fold CV accuracy, mean +/- std)  k={K_TARGET}  [{DATA_VARIANT}]")
print("=" * 60)

header = f"{'Method':<16}" + "".join(f"  {c:<22}" for c in clf_names) + "  T(s)"
print(header)
print("-" * len(header))

for fs_name in FS_METHODS:
    row  = {c: results[fs_name][c] for c in clf_names}
    line = f"{fs_name:<16}"
    for c in clf_names:
        mu, sd = np.mean(row[c]), np.std(row[c])
        line  += f"  {mu:.4f}+/-{sd:.4f}        "
    line += f"  {timing[fs_name]:.1f}"
    print(line)

print()
print("Selected feature indices per method:")
for fs_name, sel in selections.items():
    print(f"  {fs_name:<16}: {list(sel)}")


out_path = DATA_DIR / f"results_waveform_{DATA_VARIANT}.csv"
df_res = pd.DataFrame([
    {"Method":            fs_name,
     "Classifier":        clf_name,
     "Mean_Acc":          np.mean(results[fs_name][clf_name]),
     "Std_Acc":           np.std(results[fs_name][clf_name]),
     "Selected_Features": str(list(selections[fs_name])),
     "T_sel_s":           timing[fs_name]}
    for fs_name in FS_METHODS
    for clf_name in clf_names
])
try:
    df_res.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
except PermissionError:
    fallback_out_path = DATA_DIR / (
        f"results_waveform_{DATA_VARIANT}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    )
    df_res.to_csv(fallback_out_path, index=False)
    print(f"\nCould not write to {out_path} because it is in use.")
    print(f"Results saved to {fallback_out_path} instead.")
