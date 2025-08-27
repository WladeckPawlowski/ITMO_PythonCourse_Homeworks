import pandas as pd
import numpy as np

file_path = r'C:\Users\User\Desktop\train.csv'
data = pd.read_csv(file_path)

# Извлечение имен
def extract_name(full_name):
    name_part = full_name.split(',')[1]
    name = name_part.split('.')[1].strip()
    return name
data['FirstName'] = data['Name'].apply(extract_name)

# Основная информация о датасете
print("Dataset Information:")
print(data.info())

# Количество пропусков в каждом столбце
print("\nNumber of Missing Values:")
print(data.isnull().sum())

# Средние значения для числовых столбцов
print("\nMean Values:")
print(data.mean(numeric_only=True))

# Описательная статистика
print("\nDescriptive Statistics:")
print(data.describe(include='all'))

# Процент выживаемости у каждого класса пассажиров
survival_rate_by_class = data.groupby('Pclass')['Survived'].mean() * 100
print("\nSurvival Rate by Passenger Class:")
print(survival_rate_by_class)

# Самое популярное мужское и женское имя на корабле
most_popular_male_name = data[data['Sex'] == 'male']['FirstName'].mode()[0]
most_popular_female_name = data[data['Sex'] == 'female']['FirstName'].mode()[0]

print("\nMost Popular Male Name on the Ship:")
print(most_popular_male_name)
print("Most Popular Female Name on the Ship:")
print(most_popular_female_name)

# Самое популярное мужское и женское имя на корабле в каждом классе
print("\nMost Popular Male and Female Names on the Ship by Class:")
popular_names_by_class = []
for pclass in data['Pclass'].unique():
    most_popular_male_name_class = data[(data['Sex'] == 'male') & (data['Pclass'] == pclass)]['FirstName'].mode()[0]
    most_popular_female_name_class = data[(data['Sex'] == 'female') & (data['Pclass'] == pclass)]['FirstName'].mode()[0]
    popular_names_by_class.append({
        'Pclass': pclass,
        'Most Popular Male Name': most_popular_male_name_class,
        'Most Popular Female Name': most_popular_female_name_class
    })
popular_names_by_class_df = pd.DataFrame(popular_names_by_class)
print(popular_names_by_class_df)

# Пассажиры, возраст которых больше 44 лет
passengers_over_44 = data[data['Age'] > 44]
print("\nPassengers Over 44 Years Old:")
print(passengers_over_44)

# Пассажиры, возраст которых меньше 44 лет и которые мужского пола
passengers_under_44_male = data[(data['Age'] < 44) & (data['Sex'] == 'male')]
print("\nPassengers Under 44 Years Old and Male:")
print(passengers_under_44_male)

# Количества n-местных кабин (в которых было 2, 3, 4, ... человека)
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
cabin_sizes = data['FamilySize'].value_counts().sort_index().reset_index()
cabin_sizes.columns = ['FamilySize', 'Count']
print("\nCabin Sizes:")
print(cabin_sizes)

# Сохранение данных в Excel
output_file = r'C:\Users\User\Desktop\titanic_analysis.xlsx'
with pd.ExcelWriter(output_file) as writer:
    data.describe(include='all').to_excel(writer, sheet_name='Description')
    survival_rate_by_class.to_excel(writer, sheet_name='Survival Rate by Class')
    popular_names_by_class_df.to_excel(writer, sheet_name='Popular Names by Class', index=False)
    passengers_over_44.to_excel(writer, sheet_name='Passengers Over 44', index=False)
    passengers_under_44_male.to_excel(writer, sheet_name='Passengers Under 44 Male', index=False)
    cabin_sizes.to_excel(writer, sheet_name='Cabin Sizes', index=False)

print(f"Data successfully saved to file: {output_file}")
