#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, brier_score_loss


# *** globals + menu driven ***

df = None
df2 = None
df3 = None
X = None
y = None
num_feats = ['age', 'fnlwgt', 'education_num', 'capital_gain',
             'capital_loss', 'hours_per_week']

rf = None
svm_clf = None
model = None   # CNN

rf_df = None
svm_df = None
cnn_df = None
avg_table = None

auc_rf = None
auc_svm = None
auc_cnn = None

fpr_rf = tpr_rf = None
fpr_svm = tpr_svm = None
fpr_cnn = tpr_cnn = None

trained = False  # Don’t retrain every time


# ---- metrics ----
def calc_metrics(y_true, y_pred):
    # confusion matrix 
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    P = tp + fn
    N = tn + fp

    # accuracy, precision, recall, etc 
    acc = (tp + tn) / (P + N) 
    prec = tp / (tp + fp) 
    rec = tp / (tp + fn) 
    f1 = 2*prec*rec/(prec+rec)
    err = 1 - acc

    # rates
    tpr = rec
    tnr = tn / (tn + fp) 
    fpr = fp / (fp + tn) 
    fnr = fn / (tp + fn) 
    bal_acc = (tpr + tnr) / 2

    # skill scores
    tss = tpr - fpr
    hss = (2*(tp*tn - fp*fn)) / (P*(fn+tn) + N*(tp+fp)) 

    results = {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "TPR": tpr, "TNR": tnr,
        "FPR": fpr, "FNR": fnr,
        "Precision": prec,
        "Accuracy": acc,
        "Recall": rec,
        "F1_measure": f1,
        "Error_rate": err,
        "BAcc": bal_acc,
        "TSS": tss,
        "HSS": hss
    }

    # one per line
    for k, v in results.items():
        print(f"{k}: {v}")
    print()

    return results


# simple 1D CNN binary 
class CNN_mod(nn.Module):
    def __init__(self, num_features):
        
        super(CNN_mod, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding="same")
      
        # ReLU activation function to introduce non-linearity
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # fully connected layers (Dense layers)
        self.fc1 = nn.Linear(32 * num_features, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
    
        x = self.conv1(x)
        # Apply the ReLU activation function
        x = self.relu(x)   
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        # Final output layer  for binary classification
        x = self.fc2(x)
        
        return x

def prep_data(csv_path):
    global df, df2, df3, X, y

    csv_path = os.path.expanduser(csv_path)
    print(f"Loading from: {csv_path}")
    df = pd.read_csv(csv_path)
    print("Loaded.")

    # fix col names
    df.columns = df.columns.str.lower().str.replace('.', '_')

    df2 = df.replace('?', np.nan)
    df2 = df2.dropna()

    # income_binary
    df2['income_binary'] = (df2['income'] == '>50K').astype(int)

    for col in num_feats:
        df2[col] = pd.to_numeric(df2[col], errors='coerce')

    # drop original income and get dummies
    df2_no_income = df2.drop(columns=['income'])
    df3 = pd.get_dummies(df2_no_income, drop_first=True)

    X = df3.drop('income_binary', axis=1)
    y = df3['income_binary']

    print("Data prepared (df2/df3/X/y).")


def show_info():
    # Option 1: show info + plots
    global df, df2

    if df is None or df2 is None:
        print("Data not loaded yet, will load")
        prep_data(default_csv_path())

    print("\n*** Dataset Info ***")
    print("\nRaw df.info():")
    print(df.info())
    print("\nHead of df:")
    print(df.head(10))

    print("\nCleaned df2 info:")
    print(df2.info())
    print("\nIncome_binary counts:")
    print(df2['income_binary'].value_counts())

    # missing value ?
    cols = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
            'marital_status', 'occupation', 'relationship', 'race', 'sex',
            'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']

    print("\nNumber of rows with ? in important columns:")
    print(df[cols].eq('?').any(axis=1).sum())

    cols_with_q = (df == '?').any()
    print("\nColumns containing '?':")
    print(cols_with_q[cols_with_q].index.tolist())

    # ----- plots -----
    print("\nPlotting hist+KDE for numeric features...")
    fig, ax = plt.subplots(len(num_feats), 1, figsize=(12, 4 * len(num_feats)))

    for i, col in enumerate(num_feats):

        x0 = df2[df2['income_binary'] == 0][col].dropna()
        x1 = df2[df2['income_binary'] == 1][col].dropna()

        bins = np.histogram(np.hstack([x0, x1]), bins=30)[1]
        width = (bins[1] - bins[0]) * 0.45

        ax[i].hist(x0, bins=bins - width / 2, color='blue', alpha=0.8,
                   label='<=50K (0)', edgecolor='black', width=width)
        ax[i].hist(x1, bins=bins + width / 2, color='red', alpha=0.8,
                   label='>50K (1)', edgecolor='black', width=width)

        kde0 = gaussian_kde(x0)
        kde1 = gaussian_kde(x1)
        xs = np.linspace(min(bins), max(bins), 300)
        ax[i].plot(xs, kde0(xs) * len(x0) * (bins[1] - bins[0]), color='blue')
        ax[i].plot(xs, kde1(xs) * len(x1) * (bins[1] - bins[0]), color='red')

        ax[i].set_title(f"{col} distribution by income")
        ax[i].legend()

    plt.tight_layout()
    plt.show()

    # pie chart
    l = list(df2['income_binary'].value_counts())
    circle = [l[1] / sum(l) * 100, l[0] / sum(l) * 100]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.pie(circle,
           labels=['Below 50K', 'Above 50K'],
           autopct='%1.1f%%',
           startangle=90,
           explode=(0.1, 0),
           wedgeprops={'edgecolor': 'black', 'linewidth': 1, 'antialiased': True})
    ax.set_title('Income Distribution')
    plt.show()

    print("\nPlotting pairplot")
    sns.pairplot(df2, hue='income_binary')
    plt.show()


def train_models_and_kfold():
    # Runs whole training + KFold
    global X, y, rf, svm_clf, model
    global rf_df, svm_df, cnn_df, avg_table
    global auc_rf, auc_svm, auc_cnn
    global fpr_rf, tpr_rf, fpr_svm, tpr_svm, fpr_cnn, tpr_cnn
    global trained

    if X is None or y is None:
        print("Data loaded, loading now")
        prep_data(default_csv_path())

    print("\n*** Training models onece ***")

    # base split for CNN training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train[num_feats] = scaler.fit_transform(X_train[num_feats])
    X_test[num_feats] = scaler.transform(X_test[num_feats])

    # RF
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)

    # SVM
    svm_clf = SVC(
        kernel='linear',
        class_weight='balanced',
        probability=True,
        random_state=42
    )
    svm_clf.fit(X_train, y_train)

    # CNN training
    X_train_cnn = X_train.astype(np.float32)
    X_test_cnn = X_test.astype(np.float32)

    X_train_cnn_t = torch.tensor(X_train_cnn.values, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

    train_ds = TensorDataset(X_train_cnn_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    pos = y_train_t.sum()
    neg = len(y_train_t) - pos
    pos_weight = neg / pos

    print("CNN train pos:", pos.item(), "neg:", neg.item(), "pos_weight:", pos_weight.item())

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    num_features = X_train_cnn.shape[1]

    model = CNN_mod(num_features=num_features)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        print(f"Epoch {epoch} Train Loss: {np.mean(batch_losses):.4f}")

    #  KFold on full X / y 
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    all_rf_metrics = []
    all_svm_metrics = []
    all_cnn_metrics = []

    last_y_test = None
    last_y_proba_rf = None
    last_y_proba_svm = None
    last_probs_cnn = None

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        print(f"\n\n***** FOLD {fold} *****\n")

        X_train_k, X_test_k = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train_k, y_test_k = y.iloc[train_idx], y.iloc[test_idx]

        scaler_k = StandardScaler()
        X_train_k[num_feats] = scaler_k.fit_transform(X_train_k[num_feats])
        X_test_k[num_feats] = scaler_k.transform(X_test_k[num_feats])

        # RF
        rf.fit(X_train_k, y_train_k)
        y_pred_rf = rf.predict(X_test_k)
        metrics_rf = calc_metrics(y_test_k, y_pred_rf)
        y_proba_rf = rf.predict_proba(X_test_k)[:, 1]

        auc_rf_fold = roc_auc_score(y_test_k, y_proba_rf)
        bs_rf = brier_score_loss(y_test_k, y_proba_rf)
        baseline = y_train_k.mean()
        bs_ref = np.mean((y_test_k - baseline) ** 2)
        bss_rf = 1 - (bs_rf / bs_ref)

        metrics_rf["AUC"] = auc_rf_fold
        metrics_rf["Brier_score"] = bs_rf
        metrics_rf["BSS"] = bss_rf
        all_rf_metrics.append(metrics_rf)

        # SVM
        svm_clf.fit(X_train_k, y_train_k)
        y_pred_svm = svm_clf.predict(X_test_k)
        y_proba_svm = svm_clf.predict_proba(X_test_k)[:, 1]

        metrics_svm = calc_metrics(y_test_k, y_pred_svm)
        auc_svm_fold = roc_auc_score(y_test_k, y_proba_svm)
        bs_svm = brier_score_loss(y_test_k, y_proba_svm)
        baseline = y_train_k.mean()
        bs_ref = np.mean((y_test_k - baseline) ** 2)
        bss_svm = 1 - (bs_svm / bs_ref)
        metrics_svm["AUC"] = auc_svm_fold
        metrics_svm["Brier_score"] = bs_svm
        metrics_svm["BSS"] = bss_svm
        all_svm_metrics.append(metrics_svm)

        # CNN 
        model.eval()
        X_test_k_f = X_test_k.astype(np.float32)
        X_test_cnn_k = torch.tensor(X_test_k_f.values, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            logits_k = model(X_test_cnn_k).squeeze(1)
            probs_cnn_k = torch.sigmoid(logits_k).numpy()
            y_pred_cnn_k = (probs_cnn_k >= 0.5).astype(int)

        metrics_cnn = calc_metrics(y_test_k, y_pred_cnn_k)
        auc_cnn_fold = roc_auc_score(y_test_k, probs_cnn_k)
        bs_cnn = brier_score_loss(y_test_k, probs_cnn_k)
        bss_cnn = 1 - (bs_cnn / bs_ref)
        metrics_cnn["AUC"] = auc_cnn_fold
        metrics_cnn["Brier_score"] = bs_cnn
        metrics_cnn["BSS"] = bss_cnn
        all_cnn_metrics.append(metrics_cnn)

        fold_table = pd.DataFrame({
            "RF": metrics_rf,
            "SVM": metrics_svm,
            "CNN": metrics_cnn
        })
        print(f"*** Metrics for all Algorithms in Iteration {fold} ***\n")
        print(fold_table)

        # store last fold for ROC plots later
        last_y_test = y_test_k
        last_y_proba_rf = y_proba_rf
        last_y_proba_svm = y_proba_svm
        last_probs_cnn = probs_cnn_k

    iters = [f"iter{i}" for i in range(1, 11)]
    rf_df_local = pd.DataFrame(all_rf_metrics, index=iters).T
    svm_df_local = pd.DataFrame(all_svm_metrics, index=iters).T
    cnn_df_local = pd.DataFrame(all_cnn_metrics, index=iters).T

    # averages
    rf_avg = rf_df_local.mean(axis=1)
    svm_avg = svm_df_local.mean(axis=1)
    cnn_avg = cnn_df_local.mean(axis=1)

    avg_table_local = pd.DataFrame({
        "RF_avg": rf_avg,
        "SVM_avg": svm_avg,
        "CNN_avg": cnn_avg
    })

    # ROC curves for last fold
    fpr_rf_local, tpr_rf_local, _ = roc_curve(last_y_test, last_y_proba_rf)
    auc_rf_local = roc_auc_score(last_y_test, last_y_proba_rf)

    fpr_svm_local, tpr_svm_local, _ = roc_curve(last_y_test, last_y_proba_svm)
    auc_svm_local = roc_auc_score(last_y_test, last_y_proba_svm)

    fpr_cnn_local, tpr_cnn_local, _ = roc_curve(last_y_test, last_probs_cnn)
    auc_cnn_local = roc_auc_score(last_y_test, last_probs_cnn)

    # shove into globals
    rf_df = rf_df_local
    svm_df = svm_df_local
    cnn_df = cnn_df_local
    avg_table = avg_table_local

    fpr_rf, tpr_rf = fpr_rf_local, tpr_rf_local
    fpr_svm, tpr_svm = fpr_svm_local, tpr_svm_local
    fpr_cnn, tpr_cnn = fpr_cnn_local, tpr_cnn_local

    auc_rf, auc_svm, auc_cnn = auc_rf_local, auc_svm_local, auc_cnn_local

    print("\n***** AVG METRICS ACROSS 10 FOLDS *****")
    print(avg_table.round(decimals=2))

    trained = True
    print("\nTraining + KFold done.")


def show_roc(model_name):
    # Plot ROC curve RF/SVM/CNN using globals
    global fpr_rf, tpr_rf, fpr_svm, tpr_svm, fpr_cnn, tpr_cnn
    global auc_rf, auc_svm, auc_cnn

    if model_name == "rf":
        if fpr_rf is None:
            print("No RF ROC data yet.")
            return
        plt.figure()
        plt.plot(fpr_rf, tpr_rf, color='darkorange',
                 label=f'RF ROC (AUC = {auc_rf:.2f})')
    elif model_name == "svm":
        if fpr_svm is None:
            print("No SVM ROC data yet.")
            return
        plt.figure()
        plt.plot(fpr_svm, tpr_svm, color='darkorange',
                 label=f'SVM ROC (AUC = {auc_svm:.2f})')
    elif model_name == "cnn":
        if fpr_cnn is None:
            print("No CNN ROC data yet.")
            return
        plt.figure()
        plt.plot(fpr_cnn, tpr_cnn, color='darkorange',
                 label=f'CNN ROC (AUC = {auc_cnn:.2f})')
    else:
        print("Unknown model for ROC.")
        return

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(model_name.upper() + ' ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


def default_csv_path():
    # default path
    base_dir = os.path.dirname(__file__)
    return os.path.join(base_dir, "income_trimmed.csv")


def main():
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = default_csv_path()

    # load data once (no plots, no training)
    prep_data(csv_path)

    global trained

    #  ******* MENU *******
    while True:
        print("\n-------------")
        print("enter:")
        print("1: Dataset Information & Description")
        print("2: For Random Forest")
        print("3: For SVM")
        print("4: For CNN")
        print("5: Comparison of all")
        print("6: exit")
        choice = input("Your choice: ").strip()

        if choice == "1":
            show_info()

        elif choice == "2":
            if not trained:
                train_models_and_kfold()
            print("\n*** Random Forest metrics across folds ***")
            print(rf_df)
            print("\nAvg RF metrics (over folds):")
            print(avg_table['RF_avg'])
            print(f"\nLast ROC AUC for RF: {auc_rf:.4f}")
            show_roc("rf")

        elif choice == "3":
            if not trained:
                train_models_and_kfold()
            print("\n*** SVM metrics across folds ***")
            print(svm_df)
            print("\nAvge SVM metrics (over folds):")
            print(avg_table['SVM_avg'])
            print(f"\nLast ROC AUC for SVM: {auc_svm:.4f}")
            show_roc("svm")

        elif choice == "4":
            if not trained:
                train_models_and_kfold()
            print("\n*** CNN metrics across folds ***")
            print(cnn_df)
            print("\nAvg CNN metrics (over folds):")
            print(avg_table['CNN_avg'])
            print(f"\nLast ROC AUC for CNN: {auc_cnn:.4f}")
            show_roc("cnn")

        elif choice == "5":
            if not trained:
                train_models_and_kfold()
            
            print("\n RF metrics across all folds ")
            print(rf_df) # metrics as rows, iter1..iter10 as columns

            print("\n SVM metrics across all folds ")
            print(svm_df)

            print("\n CNN metrics across all folds ")
            print(cnn_df)

            print("\n Comparison of all models (avg of folds) ")
            print(avg_table.round(3))

        elif choice == "6":
            print("Exiting.")
            break

        else:
            print("Not valid, try again.")


if __name__ == "__main__":
    main()