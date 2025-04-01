import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                            roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import joblib

# 1. Загрузка данных
data = pd.read_csv('results2.csv')

# 2. Подготовка данных
X = data[['Нормированная энтропия', 'Сложность', 'Индекс Херста']]
y = data['Метка']

# Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')

# 2. Балансировка классов
data_balanced = pd.concat([pd.DataFrame(X_scaled, columns=X.columns),
                          pd.Series(y, name='Метка')], axis=1)
class_0 = data_balanced[data_balanced['Метка'] == 0]
class_1 = data_balanced[data_balanced['Метка'] == 1]

class_0_upsampled = resample(class_0, replace=True, n_samples=len(class_1), random_state=42)
balanced_data = pd.concat([class_0_upsampled, class_1])

X_balanced = balanced_data[X.columns]
y_balanced = balanced_data['Метка']

# 3. Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

# 4. Обучение модели
best_model = LogisticRegression(
    C=2.5,
    penalty='l1',
    solver='liblinear',
    class_weight='balanced',
    random_state=42
)
best_model.fit(X_train, y_train)
joblib.dump(best_model, 'best_logreg_model.pkl')

# 5. Предсказания и оценка
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Оптимальный порог по F1-score
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
optimal_idx_pr = np.argmax(f1_scores)
optimal_threshold = thresholds_pr[optimal_idx_pr]
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

# ROC-кривая и AUC
fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# 6. Оценка качества
print("\n=== Оценка модели с оптимальным порогом ===")
print(f"Оптимальный порог: {optimal_threshold:.3f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_optimal):.2f}")
print(f"ROC AUC: {roc_auc:.3f}")  # Добавлен вывод ROC AUC
print("\nClassification Report:")
print(classification_report(y_test, y_pred_optimal, zero_division=0))

# Матрица ошибок
conf_matrix = confusion_matrix(y_test, y_pred_optimal)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Патология', 'Норма'],
            yticklabels=['Патология', 'Норма'])
plt.title('Матрица ошибок (оптимальный порог)')
plt.savefig('confusion_matrix_optimal.png', dpi=300, bbox_inches='tight')
plt.show()

# Находим ближайший порог в thresholds_roc к optimal_threshold
optimal_idx_roc = np.argmin(np.abs(thresholds_roc - optimal_threshold))
optimal_idx_roc = min(optimal_idx_roc, len(fpr) - 1)  # Защита от выхода за границы

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.scatter(fpr[optimal_idx_roc], tpr[optimal_idx_roc], marker='o', color='red', label='Optimal threshold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая с оптимальным порогом')
plt.legend(loc="lower right")
plt.savefig('roc_curve_optimal.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. График распределения вероятностей
plt.figure(figsize=(10, 6))
for label in [0, 1]:
    sns.kdeplot(y_pred_proba[y_test == label], label=f'Class {label}')
plt.axvline(x=optimal_threshold, color='r', linestyle='--', label='Optimal threshold')
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.title('Распределение вероятностей по классам')
plt.legend()
plt.savefig('probability_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. Сохранение результатов
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred_optimal,
    'Probability': y_pred_proba
})
results.to_csv('model_predictions.csv', index=False)