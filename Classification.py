import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import numpy as np


def AddtoDF(Columnstr, df, frame):
    s = preprocessing.LabelEncoder()
    s.fit(frame[Columnstr].tolist())
    col = s.transform(frame[Columnstr].tolist())
    df[Columnstr] = col


def change_structure(frame):
    df = pd.DataFrame()
    AddtoDF('Station', df, frame)
    AddtoDF('Channel Type', df, frame)
    AddtoDF('Season', df, frame)
    AddtoDF('Year', df, frame)
    AddtoDF('Date', df, frame)
    AddtoDF('Day of week', df, frame)
    AddtoDF('Start_time', df, frame)
    AddtoDF('End_time', df, frame)
    AddtoDF('Length', df, frame)
    AddtoDF('Name of show', df, frame)
    AddtoDF('Name of episode', df, frame)
    AddtoDF('Genre', df, frame)
    AddtoDF('First time or rerun', df, frame)
    df['Temperature in Montreal during episode'] = frame['Temperature in Montreal during episode'].copy()
    return df


train_frame = pd.read_csv('data.csv')
x_train = change_structure(train_frame)
y_train = train_frame['Market Share_total'].copy()

test_frame = pd.read_csv('test.csv')
x_test = change_structure(test_frame)

x_train = x_train.dropna().astype(np.float32)
y_train = y_train.dropna().astype(np.float32)
x_test = x_test.dropna().astype(np.float32)

linear_regressor = LinearRegression()
linear_regressor.fit(x_train[0:533310], y_train[0:533310])
Y_pred = linear_regressor.predict(x_test)

pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 11000)

print("The predicted Market Share_total for test data is:\n")
print(Y_pred)
