# Tarea-3-Inteligencia-artificial
Programa
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

nombres_columnas = [
    'Sexo', 'Edad', 'Deuda', 'Casado', 'ClienteBancario', 'NivelEducativo',
    'Etnia', 'AñosEmpleado', 'Predeterminado', 'Empleado', 'CreditScore',
    'DriverLicence', 'Ciudadano', 'CodigoPostal', 'Ingresos', 'EstadoAprobacion'
]

data = pd.read_csv(r'C:\Users\jax\source\repos\Tarea 3 inteligencia artificial\Tarea 3 inteligencia artificial\Aprobación de creditos (1).csv', names=nombres_columnas, header=None)

print("=== ANÁLISIS DE APROBACIÓN DE TARJETAS DE CRÉDITO ===")
print("=" * 60)

print("\n1) PRIMEROS 30 REGISTROS DEL DATASET:")
print("-" * 50)
print(data.head(30))

print("\n2) RESUMEN ESTADÍSTICO E INFORMACIÓN DE LOS DATOS:")
print("-" * 50)
print("\nInformación general del dataset:")
print(data.info())
print("\nResumen estadístico:")
print(data.describe(include='all'))

print("\n3) ÚLTIMAS 17 LÍNEAS DEL DATASET:")
print("-" * 50)
print(data.tail(17))

print("\n4) VERIFICACIÓN Y SUSTITUCIÓN DE VALORES FALTANTES:")
print("-" * 50)
print("Valores únicos que podrían ser valores faltantes:")
for col in data.columns:
    unique_vals = data[col].unique()
    print(f"{col}: {unique_vals}")

data = data.replace('?', np.nan)
data = data.replace('null', np.nan)
data = data.replace('NAN', np.nan)

print("\nValores faltantes después de reemplazar '?', 'null' y 'NAN':")
print(data.isnull().sum())

print("\n5) IMPUTACIÓN DE VALORES FALTANTES:")
print("-" * 50)

columnas_numericas = ['Edad', 'Deuda', 'AñosEmpleado', 'CreditScore', 'CodigoPostal', 'Ingresos']
columnas_categoricas = [col for col in data.columns if col not in columnas_numericas and col != 'EstadoAprobacion']

for col in columnas_numericas:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

for col in columnas_numericas:
    if data[col].isnull().sum() > 0:
        mean_val = data[col].mean()
        data[col].fillna(mean_val, inplace=True)
        print(f"Imputados {data[col].isnull().sum()} valores faltantes en {col} con la media: {mean_val:.2f}")

print("\n6) IMPUTACIÓN DE VALORES FALTANTES EN COLUMNAS CATEGÓRICAS:")
print("-" * 50)

for col in columnas_categoricas:
    if data[col].isnull().sum() > 0:
        most_frequent = data[col].value_counts().index[0]
        data[col].fillna(most_frequent, inplace=True)
        print(f"  -> Imputados valores faltantes en '{col}' con: {most_frequent}")

print("\n7) VERIFICACIÓN FINAL DE VALORES FALTANTES:")
print("-" * 50)
missing_values = data.isnull().sum()
print("Número total de NaN en cada columna:")
print(missing_values)
print(f"\nTotal de valores faltantes en el dataset: {missing_values.sum()}")

if missing_values.sum() == 0:
    print("✅ Todos los valores faltantes han sido imputados exitosamente!")
else:
    print("⚠️ Aún hay valores faltantes que necesitan ser tratados")

print("\n8) CONVERSIÓN DE VALORES NO NUMÉRICOS A NUMÉRICOS:")
print("-" * 50)

le = LabelEncoder()

print("Procesando columnas:")
for col in data.columns:
    if data[col].dtype == 'object':
        print(f"  - Codificando columna '{col}' (tipo: {data[col].dtype})")
        data[col] = le.fit_transform(data[col].astype(str))
        print(f"    Nuevo tipo de datos: {data[col].dtype}")

print("\nTipos de datos después de la codificación:")
print(data.dtypes)

print("\n9) DIVISIÓN DE DATOS PARA MODELO DE REGRESIÓN LINEAL:")
print("-" * 50)

print("✅ train_test_split importado")

print("Forma original del dataset:", data.shape)
print("Columnas a descartar: índices 11 y 13")
print(f"Columna índice 11: '{data.columns[11]}'")
print(f"Columna índice 13: '{data.columns[13]}'")

data_features = data.drop([data.columns[11], data.columns[13]], axis=1)
print("Forma después de descartar columnas:", data_features.shape)

y = data[data.columns[13]]
X = data_features.values

print(f"Forma de X (características): {X.shape}")
print(f"Forma de y (objetivo): {y.shape}")

if y.dtype == 'object':
    y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

y_test = np.array(y_test)

print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"Tamaño del conjunto de prueba: {X_test.shape[0]} muestras")

print("\n10) MODELO DE REGRESIÓN LINEAL:")
print("-" * 50)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print("✅ Modelo de regresión lineal entrenado exitosamente!")
print(f"\nMétricas del modelo:")
print(f"  MSE Entrenamiento: {mse_train:.4f}")
print(f"  MSE Prueba: {mse_test:.4f}")
print(f"  R² Entrenamiento: {r2_train:.4f}")
print(f"  R² Prueba: {r2_test:.4f}")

print(f"\nCoeficientes del modelo:")
feature_names = [col for i, col in enumerate(data.columns) if i not in [11, 13]]
for i, coef in enumerate(model.coef_):
    print(f"  {feature_names[i]}: {coef:.4f}")
print(f"\nIntercept: {model.intercept_:.4f}")

print(f"\nPrimeras 10 predicciones vs valores reales (conjunto de prueba):")
print("Predicción | Real")
print("-" * 20)
for i in range(min(10, len(y_pred_test), len(y_test))):
    real_value = y_test[i]
    try:
        real_int = int(float(real_value))
    except (ValueError, TypeError):
        real_int = "N/A"
    print(f"{y_pred_test[i]:8.4f} | {real_int:4}")

print(type(y_test), getattr(y_test, 'shape', None))
print(y_test)

print("\n" + "=" * 60)
print("ANÁLISIS COMPLETADO EXITOSAMENTE")
print("=" * 60)
