import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt


# Charger le modèle pré-entraîné
model=joblib.load("BankFraud_model.pkl")

# Titre de l'application
st.title("Application de detection de fraudes Bancaires")
st.write("Une fois l'application lancée, entrez les détails d'une transaction et obtenez instantanément un score de risque indiquant si la transaction est frauduleuse ou non.")


# Choix du mode de prédiction
mode = st.radio("Choisissez le mode de prédiction", ["Fichier", "Entrée manuelle"])

if mode == "Fichier":
    st.subheader("Téléchargement de données")
    st.write("Téléchargez le fichier  contenant les données à prédire .")
    st.write("Le modele a été entrainé avec un dataset contenant les colonnes suivantes :"
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28' ,' Amount')
    uploaded_file = st.file_uploader("Choisissez un fichier", type=["csv", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Format de fichier non pris en charge. Veuillez télécharger un fichier CSV ou Excel.")  
    
        st.write("Aperçu des données :", df.head()) 

    if uploaded_file is not None:
        df=df.dropna() # Suppression des valeurs manquantes
        df=df.drop_duplicates()  # Suppression des doublons
        df=df.drop('Class', axis=1) # Suppression de la colonne cible
        
        categorical_cols = [col for col in df.select_dtypes(include=['object'])]  # Exclure la colonne 'country' de l'encodage
        le = LabelEncoder() 
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col])

        st.write("Données après prétraitement :", df.head())    

        # Bouton pour effectuer la prédiction
        if st.button("Prédire"): 
            with st.spinner("Analyse des risques en cours..."):
                time.sleep(2)  # Simule un calcul complexe           
            prediction = model.predict(df) 
            result = pd.DataFrame(prediction, columns=["Class"])
            st.write("Prédiction terminée : ✅ Transaction fiable si le modele prédit 0  / ⚠️ Transaction potentiellement frauduleusesi le modele prédit  1")   
            st.write("Resultat des prédictions :", pd.concat([df, result], axis=1, ignore_index=True))


        if st.button("Afficher les prédictions sous forme de graphique"):
            fig, ax = plt.subplots(figsize=(8, 6))
            df["Prediction"] = model.predict(df)  # Ajoute la colonne des prédictions
            sns.histplot(df, x="Amount", hue="Prediction", palette={0: "green", 1: "red"}, bins=50, ax=ax)
            st.pyplot(fig)

        #Boutton Reinitialiser
        if st.button('Effacer les données'):
            st.rerun() 
           
else:
    
    st.subheader("Entrée manuelle des données")
    # Entrée utilisateur avec toutes tes variables
    
    Titles=['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28', 'Amount']      

    input_data = []  # Liste pour stocker les entrées utilisateur
    for title in Titles:
        feature = st.number_input(f"{title} :", min_value=0.0, max_value=50000.0)
        input_data.append(feature)


    # Bouton pour effectuer la prédiction
    if st.button("Prédire"):    
        input_array = np.array([input_data])  # Convertir en tableau numpy    
        prediction = model.predict(input_array)     
        st.success(f"Prédiction : {prediction[0]}")    
        st.success(f"Prédiction terminée : {'✅ Transaction fiable' if prediction[0] == 0 else '⚠️ Transaction potentiellement frauduleuse'}")
    
    #Boutton Reinitialiser
    if st.button('Effacer les données'):
        st.rerun('input_data')  
