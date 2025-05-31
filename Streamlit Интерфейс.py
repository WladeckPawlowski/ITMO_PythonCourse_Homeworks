import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import streamlit as st

# Загрузка данных
data = pd.read_csv('realty_data.csv')

# Признаки
features = ['total_square', 'rooms', 'floor', 'lat', 'lon']

# Проверка количества строк с пропущенными значениями
print(f"Исходное количество строк: {len(data)}")
data_cleaned = data.dropna()
print(f"Количество строк после удаления NaN: {len(data_cleaned)}")

if len(data_cleaned) > 0:
    data = data_cleaned
else:
    imputer = SimpleImputer(strategy='mean')
    data[features] = imputer.fit_transform(data[features])

# Выбор признаков и целевой переменной
X = data[features]
y = data['price']

# Нормализация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Создание и обучение модели случайного леса
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test)

# Оценка модели
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Функция предсказания стоимости недвижимости
def predict_price(total_square, rooms, floor, lat, lon):
    new_data = [[total_square, rooms, floor, lat, lon]]
    new_data_scaled = scaler.transform(new_data)
    return model.predict(new_data_scaled)[0]

# Streamlit интерфейс
st.title('Прогнозирование стоимости недвижимости')

total_square = st.number_input('Общая площадь')
rooms = st.number_input('Количество комнат')
floor = st.number_input('Этаж')
lat = st.number_input('Широта')
lon = st.number_input('Долгота')

if st.button('Предсказать стоимость'):
    prediction = predict_price(total_square, rooms, floor, lat, lon)
    st.write(f'Прогнозируемая стоимость недвижимости: {prediction}')
