from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def evaluate_classification(y_true, y_pred):
    """
    모든 분류 프로젝트에서 재사용할 수 있는 최소 기능 평가 함수.
    
    Parameters
    ----------
    y_true : array-like
        실제 라벨
    y_pred : array-like
        예측 라벨

    """
    accuracy = accuracy_score(y_true, y_pred)
    report_df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).T
    cm = confusion_matrix(y_true, y_pred)

    return accuracy, report_df, cm


def show_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    """
    혼동행렬을 화면에 시각화하고, figure 객체를 반환하는 함수.

    """
    if labels is None:
        labels = sorted(list(set(y_true)))

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

    return fig


def data_view(cross_validation_scores, k_list):
    """K 값 그래프 시각화 함수"""

    fig, ax = plt.subplots(figsize=(10, 6))

    # 그래프 기본 구성
    ax.plot(k_list, cross_validation_scores)
    ax.set_xlabel("Number of K")
    ax.set_ylabel("Accuracy")
    ax.set_title("KNN Hyperparameter Tuning Results")
    ax.grid(True)

    
    plt.show()
    
    return fig


def scatter_template(df, x_col, y_col, hue_col=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        s=60,
        alpha=0.8,
        ax=ax
    )

    ax.grid(True)
    plt.show()

    return fig