import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
X = np.loadtxt('./output/SampleFeature(F).csv', delimiter=',')
y = np.concatenate((np.ones(len(X)//2), np.zeros(len(X)//2)))
clf = lgb.LGBMClassifier(
    learning_rate=0.0068,
    max_depth=-1,
)
skf = KFold(n_splits=5, shuffle=True, random_state=123)
precision_list = []
recall_list = []
f1_score_list = []
acc_list = []
mcc_list = []
tprs = []
aucs = []
spec_list = []
pr_aucs = []
mean_fpr = np.linspace(0, 1, 100)
fold_aucs = []
fold_auprs = []
for train_idx, test_idx in skf.split(X, y):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    clf.fit(X_train, y_train)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    threshold = 0.5
    y_pred = np.where(y_pred_prob > threshold, 1, 0)
    print(classification_report(y_test, y_pred))
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    fold_aucs.append(roc_auc)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    aupr = average_precision_score(y_test, y_pred_prob)
    fold_auprs.append(aupr)
    np.save(f"Y_pre{len(precision_list)}.npy", y_pred_prob)
    np.save(f"Y_test{len(precision_list)}.npy", y_test)
    fold_aucs.append(roc_auc)
    fold_auprs.append(aupr)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    accuracy = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_score_list.append(f1_score)
    acc_list.append(accuracy)
    mcc_list.append(mcc)
    spec_list.append(specificity)
print(f"Average accuracy: {np.mean(acc_list):.4f}")
print(f"Average precision: {np.mean(precision_list):.4f}")
print(f"Average f1-score: {np.mean(f1_score_list):.4f}")
print(f"Average MCC: {np.mean(mcc_list):.4f}")


