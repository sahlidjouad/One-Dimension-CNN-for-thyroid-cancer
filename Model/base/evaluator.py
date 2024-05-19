
import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc, RocCurveDisplay
import numpy as np
import tensorflow as tf


class EvalutionClassifer():

    def __init__(self, Models, validsets):
        self.Models = Models
        self.validsets = validsets

    def __predict_binary(self, model, data, thershold=0.5):
        prediction_label = self.__predict_probability(
            model, data)

        actual_label = self.__Get_actual_label(data)

        prediction = (prediction_label >= thershold)
        return actual_label, prediction

    def __predict_probability(self, model, data):
        prediction_value = model.predict(data)
        return tf.squeeze(prediction_value)

    def __Get_actual_label(self, data):
        actual_value = [x[1] for x in list(data.as_numpy_iterator())]
        return np.concatenate(actual_value)

    def Draw_ROC(self, i):
        prediction_value = self.__predict_probability(
            self.Models[i], self.validsets[i])

        actual_value = self.__Get_actual_label(self.validsets[i])

        fpr, tpr, thresholds = roc_curve(actual_value, prediction_value)
        roc_auc = auc(fpr, tpr)
        display = RocCurveDisplay(
            fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=f'estimator {i}')
        display.plot()
        return display

    def Draw_Confusion_Matrics(self, i, thershold=0.5):

        actual_label, prediction = self.__predict_binary(
            self.Models[i], self.validsets[i], thershold)

        conf_mat = confusion_matrix(actual_label, prediction)
        displ = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        displ.plot()
        return displ

    def Draw_ROC_ALL(self):
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        fig, ax = plt.subplots(figsize=(10, 10))
        fig.figwidth = 10
        for i, model, data in zip(range(len(self.Models)), self.Models, self.validsets):

            prediction_value = self.__predict_probability(model, data)
            actual_value = self.__Get_actual_label(data)

            fpr, tpr, thresholds = roc_curve(actual_value, prediction_value)
            roc_auc = auc(fpr, tpr)

            ax.plot(
                fpr,
                tpr,
                color=f"C{i}",
                label=r"ROC fold %i (AUC = %0.4f)" % (i, roc_auc),
                alpha=0.8)

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.4f $\pm$ %0.4f)" % (mean_auc, std_auc),
            alpha=0.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",)

        ax.plot(
            mean_fpr,
            mean_fpr,
            linestyle="dashed",
            label=r"Simple classifier",
            alpha=0.8
        )

        ax.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"Mean ROC curve with variability\n(Positive label tumor type)",
        )
        ax.axis("square")
        ax.legend(loc="lower right")

        return fig, ax

    def Calculate_metrics(self, metrics, i):
        predicted_label = self.__predict_probability(
            self.Models[i], self.validsets[i])
        actual_label = self.__Get_actual_label(self.validsets[i])
        result = {}
        for metric in metrics:
            calculator = metric(name=metric.__name__)
            calculator.update_state(actual_label, predicted_label)
            result[f"{calculator.name}"] = calculator.result().numpy()

        return result


def Drow(data: dict, n_epoch, model_name):
    """
    value one_dimension arrary.

    """
    xa = np.linspace(0, n_epoch, n_epoch)
    fig, ax = plt.subplots(figsize=(12, 6))
    i = 0
    for key, value in data.items():

        ax.plot(
            xa,
            value,
            color=f"C{i}",
            label=key + f" Curve ({value[-1]})"
        )
        i = i+1
    ax.set(
        xlabel="Epoch",
        title=f"Model {model_name}"
    )
    ax.grid()
    ax.legend()
    return fig, ax


def Save_fig(plotPath, model_number, column_names):
    """ 
    To save plot just call the function before plt.show() function.
    ex:
        Save_file(direction, model_number, column_names)
        plt.show()
    """
    namePlot = os.path.join(
        plotPath, f"{model_number}"+"_".join(column_names) + ".png")

    if os.path.exists(namePlot):
        raise Exception("The file is allready exists")

    plt.savefig(namePlot)


def Drow_mean(data: dict, n_epoch):
    xa = np.linspace(0, n_epoch, n_epoch)

    fig, ax = plt.subplots(figsize=(12, 6))

    i = 0
    for key, value in data.items():
        interp_data = [np.interp(xa, xa, list_data) for list_data in value]

        mean_values = np.mean(interp_data, axis=0)

        ax.plot(
            xa,
            mean_values,
            label=key + f" Curve {mean_values[-1]}",
            color=f"C{i}"
        )

        std = np.std(interp_data, axis=0)
        tprs_upper = np.minimum(mean_values + std, 1)
        tprs_lower = np.maximum(mean_values - std, 0)

        ax.fill_between(
            xa,
            tprs_lower,
            tprs_upper,
            color=f"C{i}",
            alpha=0.5,
            label=r"$\pm$ 1 std. dev.",
        )

        i = i+1

    ax.grid()
    ax.legend()

    return fig, ax
