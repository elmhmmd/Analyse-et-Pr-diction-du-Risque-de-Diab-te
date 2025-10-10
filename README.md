# 🩺 Analyse et Prédiction du Risque de Diabète

## 📋 Présentation

Ce projet vise à analyser un jeu de données sur le diabète afin d'identifier des groupes de patients à risque via des techniques de clustering (apprentissage non supervisé). Ensuite, plusieurs modèles de classification (apprentissage supervisé) sont entraînés pour prédire l'appartenance d'un patient à un groupe de risque, permettant ainsi une identification précoce et une meilleure prise en charge.

**Objectif métier :** Segmenter les patients en catégories de "risque élevé" et "risque faible" en se basant sur leurs données médicales, puis construire un modèle prédictif fiable pour automatiser cette classification. Les variables utilisées incluent :
*   Nombre de grossesses (Pregnancies)
*   Concentration en glucose (Glucose)
*   Pression artérielle (BloodPressure)
*   Épaisseur du pli cutané (SkinThickness)
*   Niveau d'insuline (Insulin)
*   Indice de masse corporelle (BMI)
*   Fonction généalogique du diabète (DiabetesPedigreeFunction)
*   Âge (Age)

## 🧑‍🔬 Pipeline & Fonctionnalités

1.  **Analyse Exploratoire des Données (EDA)**
    *   Importation et exploration initiale du jeu de données.
    *   Visualisation des distributions, des relations et des corrélations.
    *   Identification des zéros comme valeurs manquantes implicites dans plusieurs colonnes critiques (`Glucose`, `Insulin`, etc.).

2.  **Prétraitement des Données**
    *   **Gestion des valeurs manquantes :** Remplacement des zéros par `NaN`, puis imputation à l'aide de l'algorithme `KNNImputer` pour une estimation plus précise.
    *   **Détection et traitement des valeurs aberrantes (outliers) :** Visualisation avec `boxplot` et application d'une méthode de plafonnement (capping) basée sur l'écart interquartile (IQR) pour limiter leur influence.
    *   **Standardisation :** Mise à l'échelle des variables numériques avec `StandardScaler` pour que les modèles ne soient pas biaisés par les différentes échelles de valeurs.

3.  **Segmentation Non Supervisée (Clustering)**
    *   **Choix de l'algorithme :** Utilisation de `K-Means` pour partitionner les données.
    *   **Détermination du nombre de clusters :** Application des méthodes du coude (Elbow Method) et de la silhouette pour identifier le nombre optimal de groupes (k=2).
    *   **Visualisation et Interprétation :** Réduction de la dimensionnalité avec `PCA` pour visualiser les clusters et analyse des centroïdes pour caractériser les profils de chaque groupe ("risque élevé" vs "risque faible").

4.  **Entraînement des Modèles de Classification**
    *   **Préparation :** Division des données en ensembles d'entraînement et de test.
    *   **Gestion du déséquilibre des classes :** Utilisation de `RandomOverSampler` pour s'assurer que les modèles sont entraînés sur un nombre équilibré d'exemples de chaque classe.
    *   **Implémentation de plusieurs modèles :**
        *   `LogisticRegression`
        *   `DecisionTreeClassifier`
        *   `RandomForestClassifier`
        *   `SVC` (Support Vector Classifier)
        *   `GradientBoostingClassifier`
        *   `XGBClassifier`

5.  **Évaluation et Comparaison**
    *   **Métriques :** Évaluation de chaque modèle à l'aide de la précision, du rappel (recall), du F1-score et de la matrice de confusion.
    *   **Synthèse :** Création d'un tableau récapitulatif pour comparer les performances des modèles. La **Régression Logistique** et le **SVM** se sont révélés être les plus performants.

6.  **Test, Export et Déploiement**
    *   **Export du modèle** finalisé (via `joblib`).
    *   **Interface utilisateur simple** pour tester une prédiction à partir de données saisies.

7.  **Documentation et Reproductibilité**
    *   **Code commenté** et étapes détaillées dans le notebook.
    *   **Instructions claires d’exécution** dans ce README.

## 🚀 Comment exécuter le projet

1.  **Cloner le dépôt**
    ```bash
    git clone https://github.com/elmhmmd/Pr-dicteur-de-Charges-d-Assurance-Maladie
    cd Pr-dicteur-de-Charges-d-Assurance-Maladie
    ```

2.  **Installer les dépendances**
    Assurez-vous d'avoir Python installé. Ensuite, installez les bibliothèques nécessaires :
    ```bash
    pip install numpy pandas seaborn scikit-learn xgboost matplotlib imbalanced-learn
    ```

3.  **Lancer le notebook Jupyter**
    Le code est structuré dans un notebook Jupyter. Lancez-le pour suivre toutes les étapes de l'analyse, du prétraitement et de l'entraînement des modèles.
    ```bash
    jupyter notebook Diabetes.ipynb
    ```

4.  **Lancer l'interface Streamlit**
    Pour une démonstration interactive, lancez l'application Streamlit.
    ```bash
    streamlit run Interface.py
    ```
