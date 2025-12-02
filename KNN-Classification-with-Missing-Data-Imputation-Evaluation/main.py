from src.save_utils import *
from src.knn_model import *
from src.ml_eval_visual import *

def main():
    # train, test, 나누기 전 파일 열기
    train, test, main = file_open()

    # 자료 분포 그래프 그리기
    main["vote_rate"] = main["votes"] / main["ballots"]
    fig = scatter_template(main, "votes", "vote_rate", hue_col="inducted")

    save_png(fig, prefix="hof_votes_distribution", folder="plot")

    # K 리스트 만들기
    k_list = make_k(train)

    # 최적의 k 찾기
    best_k, cross_validation_scores = find_k(train, k_list)

    # k 그래프 그리기
    graph = data_view(cross_validation_scores, k_list)

    save_png(graph, prefix="knn_k_performance", folder="plot")

    # 모델 학습, 모델 테스트
    pred, y_test = model_test(train, test, best_k)


    print(f"\n🎯accuracy : {accuracy_score(y_test, pred):.3f}")
    print()
    print(classification_report(y_test, pred, digits=3))
    print(f"\n🔥 Best K: {best_k}")
    print()

    cm_fig = show_confusion_matrix(y_test, pred, title="KNN Confusion Matrix")

    save_png(cm_fig, prefix="knn_confusion_matrix", folder="plot")


if __name__ == "__main__":
    main()