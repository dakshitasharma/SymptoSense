

import pandas as pd
import numpy as np

df = pd.read_csv("Training.csv")


# Remove duplicate rows
df = df.drop_duplicates()

df.head(4) # 1 means yes , 0 means no

#df.shape # rows , cols

df["prognosis"].unique()

df["prognosis"].unique().shape

"""# split dataset for train and testing"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# selecting lebel and features

x = df.drop("prognosis" , axis = 1)
y = df["prognosis"]

y

# converting string to np array

label = LabelEncoder()

label.fit(y)
y_array = label.transform(y)

# y_array

# split data into 70:30
x_train , x_test , y_train , y_test =  train_test_split(x , y_array , test_size= 0.3, random_state= 42)

print(f"x_train--{x_train.shape}")
print(f"y_test--{x_test.shape}")
print(f"y_train--{y_train.shape}")
print(f"y_test--{y_test.shape}")

"""# model training"""

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import accuracy_score , mean_squared_error , precision_score , f1_score , recall_score

# KNeighborsClassifier:
# accuracy: 92.39%
# RMSE: 1.964356294609996

# GaussianNB
# accuracy: 100.00%
# RMSE: 0.0

# SVC:
# accuracy: 95.65%
# RMSE: 2.776062272199677

# LogisticRegression
# accuracy: 100.00%
# RMSE: 0.0

# I have used this models but to get better result I have used two models combinations:
# GaussianNB and LogisticRegression

# GaussianNB + LogisticRegression

gnb = GaussianNB()
lr = LogisticRegression(random_state= 42)

voting_clf = VotingClassifier(estimators=[('gnb', gnb), ('lr', lr)], voting='soft')

voting_clf.fit(x_train, y_train)

y_pred = voting_clf.predict(x_test)

accuracy = accuracy_score(y_test , y_pred)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"accuracy: {accuracy * 100:.2f}%")

print(f"RMSE: {rmse}")

precision = precision_score(y_test, y_pred, average='micro')
print(f"Precision: {precision:.2f}")

f1 = f1_score(y_test, y_pred, average='micro')
print(f"F1 Score: {f1:.2f}")

recall = recall_score(y_test, y_pred, average='micro')
print(f"Recall: {recall:.2f}")

my_model = voting_clf

import pickle
pickle.dump(my_model,open('model.pkl','wb'))

my_model = pickle.load(open('model.pkl','rb'))

x_test.iloc[0].values.reshape(1 , -1)

print("model prediction label: " , my_model.predict(x_test.iloc[0].values.reshape(1 , -1)))
print("original label: " , y_test[0])

"""# recommendation and prediction

# load datasets
"""

symtoms_df = pd.read_csv("symtoms_df.csv")
symtoms_df.head(4)

precautions_df = pd.read_csv("precautions_df.csv")
precautions_df.head(4)

workout_df = pd.read_csv("workout_df.csv")
workout_df.head(4)

description_df = pd.read_csv("description.csv")
description_df.head(4)

medications_df = pd.read_csv("medications.csv")
medications_df.head(4)

diets_df = pd.read_csv("diets.csv")
diets_df.head(4)

doctor_df = pd.read_csv("Doctor.csv")
doctor_df.head(4)

unique_values = []
unique_values2 = []

disease_dict = {}

for item in y_array:
    if item not in unique_values:
        unique_values.append(item)

for i in df["prognosis"].unique() :
    if i not in unique_values2:
        unique_values2.append(i)

for i in range (0 , len(unique_values2)) :
    disease_dict.update({unique_values[i] : unique_values2[i]})

# 41 disease
print(disease_dict)

unique_values3 = []
unique_values4 = []

symptoms_dict = {}

for item in df.columns:
    if item not in unique_values3:
        if item != "prognosis":
            unique_values3.append(item)

for i in range (0 , len(df.columns)) :
    if i not in unique_values4:
        unique_values4.append(i)

for i in range(len(df.columns)-1) :
    symptoms_dict.update({unique_values3[i] : unique_values4[i]})

# 132 cols
print(symptoms_dict)

"""# predict disease"""

# create a funtion to predict disease by using symtoms

def get_prediction(symptoms) :
    input_vector = np.zeros(len(symptoms_dict))

    for item in symptoms:
        input_vector[symptoms_dict[item]] = 1

    return disease_dict[my_model.predict([input_vector])[0]]

print("seperated your symptoms by using comma(,)")
# symptoms = input("Enter your symptoms: ")
symptoms = "skin_rash , itching"
symptoms

symptoms = symptoms.lower()

# split by comma
patient_symptoms = [s.strip() for s in symptoms.split(',')]

patient_symptoms = [sym.strip() for sym in patient_symptoms]
# patient_symptoms

predicted_disease = get_prediction(patient_symptoms)

print("predicted_disease: " , predicted_disease)

# create an helper funtion to show all the realated info of that disease
# return required columns from each datasets

def helper(disease) :

    # description

    desc = description_df[description_df["Disease"] == disease]["Description"]
    # desc

    # showing description in proper format
    desc = " ".join([i for i in desc])

#     # precaution

    precaution = precautions_df[precautions_df["Disease"] == disease][
        ["Precaution_1" , "Precaution_2" , "Precaution_3" , "Precaution_4"]
    ]
    precaution = [i for i in precaution.values]

    # symptoms

    symptoms = symtoms_df[symtoms_df["Disease"] == disease][
        ["Symptom_1" , "Symptom_2" , "Symptom_3" , "Symptom_4"]
    ]
    symptoms = [i for i in symptoms.values]

#     # medication

    medicine = medications_df[medications_df["Disease"] == disease]["Medication"]
    medicine = [i for i in medicine.values]

#     # diet

    diet = diets_df[diets_df["Disease"] == disease]["Diet"]
    diet = [i for i in diet.values]

#     # workout

    workout = workout_df[workout_df["disease"] == disease]["workout"]

#     doctor

    doctor = doctor_df[doctor_df["Disease"] == disease]

#     doctor_id =
#     Doctor ID


    return desc , precaution , medicine , diet , workout , symptoms , doctor

desc , precaution , medicine , diet , workout , symptoms , doctor = helper(predicted_disease)

print("Description:")
print(desc)

print("Precautions:")

j = 1

for i in precaution[0]:
    print(j , ": " , i)
    j += 1

print("Related symptoms:")

j = 1

for i in symptoms[0]:
    print(j , " : " , i)
    j += 1

print("Medications:")

for i in medicine:
    print(i)

print("Diets:")

for i in diet:
    print(i)

print("Workouts:")

j = 1;

for i in workout:
    print(j , ": " , i)
    j += 1

for i in doctor.Specialization:
    print("Specialization:" , i)

for i in doctor.Gender:
    print("Gender:" , i)

for i in doctor.DoctorID:
    print("DoctorID:" , i)

