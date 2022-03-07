from sklearn.metrics import classification_report


def analysis_report_mfcadplus(y_true, y_pred):
    target_names = ["Chamfer", "Through hole", "Triangular passage", "Rectangular passage", "6-sides passage",
                    "Triangular through slot", "Rectangular through slot", "Circular through slot",
                    "Rectangular through step", "2-sides through step", "Slanted through step", "O-ring", "Blind hole",
                    "Triangular pocket", "Rectangular pocket", "6-sides pocket", "Circular end pocket",
                    "Rectangular blind slot", "Vertical circular end blind slot", "Horizontal circular end blind slot",
                    "Triangular blind step", "Circular blind step", "Rectangular blind step", "Round", "Stock"]
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

    print(classification_report(y_true, y_pred, labels=labels, target_names=target_names, digits=4))


def analysis_report_mfcad(y_true, y_pred):
    target_names = ["Chamfer", "Triangular passage", "Rectangular passage", "6-sides passage", "Triangular through slot",
                    "Rectangular through slot", "Rectangular through step", "2-sides through step", "Slanted through step",
                    "Triangular pocket", "Rectangular pocket", "6-sides pocket", "Rectangular blind slot",
                    "Triangular blind step", "Rectangular blind step", "Stock"]

    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    print(classification_report(y_true, y_pred, labels=labels, target_names=target_names, digits=4))
