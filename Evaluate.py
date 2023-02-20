

from imblearn.metrics import geometric_mean_score, specificity_score
from sklearn.metrics import confusion_matrix

def Evaluate(model, data, labels):

    prediction = (model.predict(data) > 0.5).astype("int32")
    specificity = specificity_score(labels, prediction)
    gmean = geometric_mean_score(labels, prediction)
    score = model.evaluate(data, labels, verbose=0)
    tn, fp, fn, tp = confusion_matrix(labels, prediction).ravel()

    results = {"Loss": score[0],
               "Acc": score[1],
               "AUC": score[4],
               "GMean": gmean,
               "Recall": score[3],
               "Precision": score[2],
               "Spec": specificity,
               "True Positives": tp,
               "True Negatives": tn,
               "False Positives": fp,
               "False Negatives": fn}

    return results
