from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка данных
data = pd.read_csv(r'C:\Users\User\Desktop\Model\realty_data.csv')

# Определение признаков
features = ['total_square', 'rooms', 'floor', 'lat', 'lon']

# Обработка пропущенных значений
imputer = SimpleImputer(strategy='mean')
data[features] = imputer.fit_transform(data[features])

# Нормализация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[features])

# Обучение модели
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, data['price'])

class Item(BaseModel):
    total_square: float
    rooms: int
    floor: int
    lat: float
    lon: float

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/predict_get")
def predict_get(
    total_square: float = Query(..., description="Total Square"),
    rooms: int = Query(..., description="Number of Rooms"),
    floor: int = Query(..., description="Floor"),
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude")
):
    new_data = [[total_square, rooms, floor, lat, lon]]
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)
    return {"prediction": prediction[0]}

@app.post("/predict_post")
def predict_post(item: Item):
    new_data = [[item.total_square, item.rooms, item.floor, item.lat, item.lon]]
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)
    return {"prediction": prediction[0]}

# Запуск приложения
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
