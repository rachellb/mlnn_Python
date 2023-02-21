


def averageResults(Results):



    sum(d['GMean'] for d in Results) / len(Results)

    results = {"GMean": sum(d["GMean"] for d in Results) / len(Results),
               "Acc": sum(d["Acc"] for d in Results) / len(Results),
               "AUC": sum(d["AUC"] for d in Results) / len(Results),
               "Recall": sum(d["Recall"] for d in Results) / len(Results),
               "Spec": sum(d["Spec"] for d in Results) / len(Results)}

    return results
