# KNN for Baseball Hall of Fame Classification

## (결측치 처리 문제 해결 실험)

---

## 📅 Date: 2025-11-28

---

## 🎯 Objective

- 결측치를 **0으로 대체한 오류**로 인해 KNN 성능이 왜곡된 문제 발견
- 결측치 행을 **완전 제거 후** 동일한 실험을 재수행하여
- 데이터 전처리의 중요성과 성능 향상을 검증

---

## ⚠ Challenge

- 득표 데이터 결측치 → “0표”로 오해
- 과거 헌액자(Y)가 (0,0) 영역에 뭉치는 **거짓 패턴 학습**
- KNN이 **0표 → 헌액(Y)** 로 잘못 판단

---

## 🔧 Approach

- 결측치 행 제거, 득표율 재계산
- Stratified Train/Test Split (8:2)
- KNN Hyperparameter Tuning (Grid Search)
- 평가 지표: Accuracy + Precision/Recall/F1

---

## 🏁 Key Achievements

- 데이터 오류 제거 후 **정상적인 패턴 학습**
- 헌액자(Y) Recall 1.0 → **한 명도 놓치지 않음**
- Accuracy: **92%** (재실험 후 성능 상승)
- 데이터 전처리의 **중요성 체득**

---

## 🧠 Model: KNN (k=51)

---

## 📌 Dataset Info

- samples: 87
    
    (결측치 제거 후)
    
- class balance: N: 68 / Y: 19

---

## 📦 Hyperparameters

- K range: 1~100 Grid Search
- Best K: **51**
- Distance Metric: Euclidean

---

## 📊 Performance (Summary)

| class | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| **N** | 1.00 | 0.897 | 0.946 | 68 |
| **Y** | 0.731 | 1.00 | 0.844 | 19 |
| **Accuracy** |  |  | **0.920** | **87** |

---

## 📄 Classification Report (Raw Text)

```bash
Accuracy: 0.9195
Y Recall: 1.00
F1(N/Y): 0.946 / 0.844

```

---

## 📈 Saved Visuals

- knn_accuracy_plot.png
- knn_confusion_matrix.png
- hof_votes_vote_rate.png

---

## 🚀 실행 방법

```bash
git clone <https://github.com/Rohstar0613/knn-baseball-classification>
cd knn-baseball-classification
pip install -r requirements.txt
python main.py

```

---

## 🧠 More Details & Reflection

👉 전처리 오류 발견 과정과 재실험 기록

👉 데이터와 모델 관계 이해 중요성

🔗 https://rohstar.tistory.com/entry/KNN-%EB%B6%84%EB%A5%98-%EC%8B%A4%ED%97%98-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EA%B2%B0%EC%B8%A1%EC%B9%98-%EC%B2%98%EB%A6%AC-%EB%AC%B8%EC%A0%9C-%ED%95%B4%EA%B2%B0


>
