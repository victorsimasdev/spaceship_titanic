import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.fillna({'HomePlanet': train['HomePlanet'].mode()[0]}, inplace=True)
train.fillna({'CryoSleep': train['CryoSleep'].mode()[0]}, inplace=True)
train.fillna({'Destination': train['Destination'].mode()[0]}, inplace=True)
train.fillna({'VIP': train['VIP'].mode()[0]}, inplace=True)
train.fillna({'Name': 'Desconhecido'}, inplace=True)
train.fillna({'Cabin': 'Desconhecido'}, inplace=True)
train.fillna({'RoomService': 0}, inplace=True)
train.fillna({'FoodCourt': 0}, inplace=True)
train.fillna({'ShoppingMall': 0}, inplace=True)
train.fillna({'Spa': 0}, inplace=True)
train.fillna({'VRDeck': 0}, inplace=True)
train.fillna({'Age': train['Age'].median()}, inplace=True)

print(train.isnull().sum())