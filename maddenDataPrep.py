#Import Library
import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing
import sklearn
from sklearn.utils import shuffle
import pickle
import csv

matchData = pd.read_csv("matchData.csv", sep=",")
teamData = pd.read_csv("teamPositions.csv", sep=",")

masterMatrix = []

fieldsT = []
rowsT = []
teamStats = {}
with open("teamPositions.csv", 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fieldsT = next(csvreader)
    for row in csvreader:
        rowsT.append(row)
        teamStats[row[0]] = row[1:]

fieldsM = []
rowsM = []
with open("matchData.csv", 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fieldsM = next(csvreader)
    for row in csvreader:
        rowsM.append(row)

teamBook = {}
for team in range(len(teamData["Team"])):
    teamBook[teamData["Team"][team]] = team

for matchup in rowsM:
    masterMatrix.append(teamStats[matchup[0]] + teamStats[matchup[1]] + [teamBook[matchup[2]]])

away = []
home = []

for pos in teamData:
    if(pos == "Team"):
        continue
    away.append("AWAY " + pos)
    home.append("HOME " + pos)


combined = away + home + ["Winner"]
df = pd.DataFrame(masterMatrix, columns = combined)

predict = "Winner"
x = np.array(df.drop([predict], 1))
y = np.array(df[predict])
print(df)

# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
best = 0
for _ in range(1):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    
    linear = linear_model.LinearRegression()
    
    linear.fit(x_train, y_train)
    predictions = linear.predict(x_test)

    for x in range(len(predictions)):
        print(predictions[x], x_test[x], y_test[x])
    acc = linear.score(x_test, y_test)
    print("Accuracy: " + str(acc))

    if acc > best:
        best = acc
        print("Accuracy: " + str(acc))
        with open("studentgrades.pickle", "wb") as f:
            pickle.dump(linear, f)
