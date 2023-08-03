import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, accuracy_score, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

X = np.loadtxt('SampleFeature.csv', delimiter=',')

y = np.concatenate((np.ones(len(X)//2), np.zeros(len(X)//2)))


clf = lgb.LGBMClassifier()


skf = StratifiedKFold(n_splits=5)
precision_list = []
recall_list = []
f1_score_list = []
acc_list = []
mcc_list = []
tprs = []
aucs = []
pr_aucs = []
mean_fpr = np.linspace(0, 1, 100)

with open("5-fold data.txt", "w") as f:
    f.write("")
    f.write(f"\t\tAccuracy\tPrecision\tRecall\tF1-score\tMCC\n")

for train_idx, test_idx in skf.split(X, y):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    clf.fit(X_train, y_train)

    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    threshold = 0.5
    y_pred = np.where(y_pred_prob > threshold, 1, 0)

    print(classification_report(y_test, y_pred))

    np.save(f"Y_pre{len(precision_list)}.npy", y_pred_prob)
    np.save(f"Y_test{len(precision_list)}.npy", y_test)

    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    accuracy = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_score_list.append(f1_score)
    acc_list.append(accuracy)
    mcc_list.append(mcc)


    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1_score:.4f}")
    print(f"MCC: {mcc:.4f}")
    with open("5-fold data.txt", "a") as f:

        f.write(f"\t\t{accuracy:.4f}\t  {precision:.4f}\t  {recall:.4f}\t  {f1_score:.4f}\t  {mcc:.4f}\n")

print(f"Average accuracy: {np.mean(acc_list):.4f}")
print(f"Average precision: {np.mean(precision_list):.4f}")
print(f"Average recall: {np.mean(recall_list):.4f}")
print(f"Average f1-score: {np.mean(f1_score_list):.4f}")
print(f"Average MCC: {np.mean(mcc_list):.4f}")

with open("5-fold data.txt", "a") as f:
    f.write(f"Average: {np.mean(acc_list):.4f}\t {np.mean(precision_list):.4f}\t {np.mean(recall_list):.4f}\t {np.mean(f1_score_list):.4f}\t {np.mean(mcc_list):.4f}\n")
