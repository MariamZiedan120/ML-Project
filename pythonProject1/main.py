import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('Training.csv')
X = df.drop("prognosis", axis=1)
y = df['prognosis']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
label_encoder = LabelEncoder()

# Fit and transform the target variable
y_encoded = label_encoder.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Initialize the Logistic Regression model
lr_model = LogisticRegression()

# Train the model on the training set
lr_model.fit(x_train, y_train)

# Make predictions on the training set
y_train_pred_lr = lr_model.predict(x_train)
# Make predictions on the testing set
y_test_pred_lr = lr_model.predict(x_test)

# Evaluate the model
print("Training Accuracy:", accuracy_score(y_train, y_train_pred_lr))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred_lr))
# Display confusion matrix and classification report for testing set
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred_lr))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred_lr))
from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest model
rf_model = RandomForestClassifier()

# Train the model on the training set
rf_model.fit(x_train, y_train)

# Make predictions on the training set
y_train_pred_rf = rf_model.predict(x_train)
# Make predictions on the testing set
y_test_pred_rf = rf_model.predict(x_test)

# Evaluate the model
print("Training Accuracy:", accuracy_score(y_train, y_train_pred_rf))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred_rf))

# Display confusion matrix and classification report for testing set
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred_rf))
from sklearn.svm import SVC

# Initialize the Support Vector Classifier model
svc_model = SVC()

# Train the model on the training set
svc_model.fit(x_train, y_train)

# Make predictions on the training set
y_train_pred_svc = svc_model.predict(x_train)
# Make predictions on the testing set
y_test_pred_svc = svc_model.predict(x_test)

# Evaluate the model
print("Training Accuracy:", accuracy_score(y_train, y_train_pred_svc))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred_svc))

# Display confusion matrix and classification report for testing set
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred_svc))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred_svc))
from sklearn.naive_bayes import GaussianNB

# Initialize the Gaussian Naive Bayes model
nb_model = GaussianNB()

# Train the model on the training set
nb_model.fit(x_train, y_train)

# Make predictions on the training set
y_train_pred_nb = nb_model.predict(x_train)
# Make predictions on the testing set
y_test_pred_nb = nb_model.predict(x_test)

# Evaluate the model
print("Training Accuracy:", accuracy_score(y_train, y_train_pred_nb))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred_nb))

# Display confusion matrix and classification report for testing set
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred_nb))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred_nb))
from sklearn.ensemble import GradientBoostingClassifier

# Initialize the Gradient Boosting Classifier model
gb_model = GradientBoostingClassifier()

# Train the model on the training set
gb_model.fit(x_train, y_train)

# Make predictions on the training set
y_train_pred_gb = gb_model.predict(x_train)
# Make predictions on the testing set
y_test_pred_gb = gb_model.predict(x_test)

# Evaluate the model
print("Training Accuracy:", accuracy_score(y_train, y_train_pred_gb))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred_gb))

# Display confusion matrix and classification report for testing set
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred_gb))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred_gb))
models = {
    'Logistic Regression': lr_model,
    'Random Forest': rf_model,
    'Support Vector Classifier': svc_model,
    'Naive Bayes': nb_model,
    'Gradient Boosting Classifier': gb_model
}

# Initialize a dictionary to store accuracies
accuracies = {}

# Iterate over models
for name, model in models.items():
    # Make predictions on the testing set
    y_pred = model.predict(x_test)
    # Calculate accuracy and store it
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc

# Print accuracies
for name, acc in accuracies.items():
    print(f'{name} Accuracy:', acc)
    import pickle

    # Save the trained SVC model
    with open('svc_model.pkl', 'wb') as file:
        pickle.dump(svc_model, file)

    print("SVC model saved successfully.")
    import pickle

    # Load the saved SVC model
    with open('svc_model.pkl', 'rb') as file:
        loaded_svc_model = pickle.load(file)

    print("SVC model loaded successfully.")

symtoms = pd.read_csv("symtoms_df.csv")
precautions = pd.read_csv('precautions_df.csv')
workout = pd.read_csv('workout_df.csv')
description = pd.read_csv('description.csv')
medications = pd.read_csv('medications.csv')
diets = pd.read_csv('diets.csv')
from tkinter import  *
from tkinter import  messagebox


root = Tk()
root.maxsize(800,800)
root.configure(width="600",height="600",bg="lightblue")
root.minsize(200,200)
root.title("disease prediction")
label = Label(root , text="disease prediction project ")
label.configure(bg="Lightblue" , foreground="white" , font=("Arial" ,20 , "bold"))
label.pack()
label2 = Label(root , text="Enter a symptom (or type 'done' to finish):")
label2.configure(bg="blue" , foreground="white" , font=("Arial" ,15 , "bold"))
label2.pack()
my_entry = Entry(root, width=40 , foreground="white" , bg = "gray")
my_entry.pack(pady=10)
symptoms = []
def predict_disease():
    symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4,
                 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9,
                 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13,
                 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18,
                 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22,
                 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27,
                 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
                 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37,
                 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42,
                 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46,
                 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50,
                 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54,
                 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58,
                 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61,
                 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66,
                 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70,
                 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74,
                 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78,
                 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82,
                 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86,
                 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89,
                 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92,
                 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96,
                 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100,
                 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103,
                 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107,
                 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110,
                 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113,
                 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116,
                 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120,
                 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124,
                 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127,
                 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
    diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction',
                 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma',
                 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)',
                 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A',
                 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis',
                 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)',
                 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism',
                 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis',
                 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection',
                 35: 'Psoriasis', 27: 'Impetigo'}
    num_features = 132  # Adjust this based on the actual number of features expected by svc_model
    feature_vector = [0] * num_features

    # Encode symptoms in the feature vector
    for symptom in symptoms:
        if symptom in symptoms_dict:
            feature_vector[symptoms_dict[symptom]] = 1
    prediction = svc_model.predict([feature_vector])
    predicted_disease = diseases_list[prediction[0]]

    predicted_description_info = description[description['Disease'] == predicted_disease]['Description'].values
    # Extract the symptoms information for the predicted disease and remove duplicates
    predicted_symptoms_info = symtoms[symtoms['Disease'] == predicted_disease][
        ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']].values
    predicted_symptoms_info = pd.unique(predicted_symptoms_info.ravel())

    # Extract the medications information for the predicted disease
    predicted_medications_info = medications[medications['Disease'] == predicted_disease]['Medication'].values
    # Extract the precautions information for the predicted disease and remove duplicates
    predicted_precautions_info = precautions[precautions['Disease'] == predicted_disease][
        ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values
    predicted_precautions_info = pd.unique(predicted_precautions_info.ravel())

    # Extract the diet information for the predicted disease
    predicted_diet_info = diets[diets['Disease'] == predicted_disease]['Diet'].values
    predicted_workout_info = workout[workout['disease'] == predicted_disease]['workout'].values
    res = "your disease is " + str(predicted_disease) + "\n your disease description is :  " + str(predicted_description_info) + "\n Symptoms : " + str(predicted_symptoms_info) + " \n Medications : " + str(predicted_medications_info) + "\n Precautions : " + str(predicted_precautions_info) + "\n Diet :" + str(predicted_diet_info) + "\n Workout" + str(predicted_workout_info)
    messagebox.showinfo("Detection", res)
def collect_symptoms():
    symptom = my_entry.get().strip().lower()
    symptoms.append(symptom)

my_button= Button(text="add_symptom",command=collect_symptoms,bg="black",foreground="white",activebackground="gray")
my_button.pack(pady=10)
my_button2= Button(text="Detect",command=predict_disease,bg="black",foreground="white",activebackground="gray")
my_button2.pack(pady=10)
root.mainloop()


