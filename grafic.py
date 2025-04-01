import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Загрузка данных
data = pd.read_csv('results.csv')

# Разделение данных по меткам
class_0 = data[data['Метка'] == 0]
class_1 = data[data['Метка'] == 1]

# Создание 3D графика
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Отображение точек класса 0 (красные)
ax.scatter(
    class_0['Нормированная энтропия'],
    class_0['Сложность'],
    class_0['Индекс Херста'],
    c='red',
    label='Class 0',
    s=50,
    alpha=0.7,
    edgecolors='w'
)

# Отображение точек класса 1 (синие)
ax.scatter(
    class_1['Нормированная энтропия'],
    class_1['Сложность'],
    class_1['Индекс Херста'],
    c='blue',
    label='Class 1',
    s=50,
    alpha=0.7,
    edgecolors='w'
)

# Настройка осей и заголовка
ax.set_xlabel('Нормированная энтропия', fontsize=12)
ax.set_ylabel('Сложность', fontsize=12)
ax.set_zlabel('Индекс Херста', fontsize=12)
ax.set_title('3D Визуализация данных по классам', fontsize=16)

# Легенда
ax.legend(fontsize=12)

# Оптимальный угол обзора
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.show()