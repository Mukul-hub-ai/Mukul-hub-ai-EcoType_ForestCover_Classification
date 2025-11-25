# Mukul-hub-ai-EcoType_ForestCover_Classification
Forest Cover Classification uses machine learning to predict 7 forest types from terrain data. Implements 5 algorithms with KNN achieving best performance after hyperparameter tuning. Features Streamlit web app for real-time predictions, aiding environmental research and forest management.
Forest_Cover_Classification
Forest Cover Classification is a comprehensive machine learning project dedicated to predicting forest cover types based on cartographic variables. This project integrates advanced data preprocessing, feature engineering, multiple classification algorithms, and Streamlit web deployment to deliver accurate predictions for environmental research and forest management.

Architecture Description
The architecture of the Forest Cover Classification project follows a robust ML pipeline, integrating Python, Scikit-learn, and Streamlit to deliver predictive insights from terrain and environmental data.

Step 1: Data Extraction & Preprocessing
Raw forest cover data is loaded from CSV files containing 581,012 observations

54 original features including elevation, slope, aspect, hydrology distances, and wilderness areas

Comprehensive data cleaning including outlier detection and treatment using IQR method

Missing value analysis and data quality checks

Step 2: Data Cleaning & Feature Engineering
The raw data is processed using pandas and numpy in Jupyter Notebooks

Outliers are capped using Interquartile Range (IQR) method

Skewness treatment using log transformation for highly skewed features

New features are engineered:

Hydrology_Distance_Ratio (Vertical/Horizontal hydrology distance)

Road_to_Hydrology_Ratio (Roadways to hydrology distance)

Fire_to_Hydrology_Ratio (Fire points to hydrology distance)

Hillshade_Morning_Noon_Diff (Light intensity morning variation)

Hillshade_Noon_Evening_Diff (Light intensity afternoon variation)

Hillshade_Daily_Variation (Total daily light variation)

Step 3: Feature Selection & Model Training
Random Forest feature importance used to select top 20 most predictive features

Multiple classification algorithms implemented and evaluated:

Random Forest Classifier

Decision Tree Classifier

Logistic Regression

K-Nearest Neighbors

XGBoost Classifier

Comprehensive model evaluation using accuracy, precision, recall, confusion matrix

Step 4: Hyperparameter Tuning & Optimization
Best performing model (K-Nearest Neighbors) selected for hyperparameter tuning

RandomizedSearchCV with 5-fold cross-validation for optimal parameter discovery

Parameters tuned: n_neighbors, weights, algorithm, leaf_size, distance metric

Final model saved for deployment using joblib/pickle

Step 5: Streamlit Web Application
Interactive web application built using Streamlit framework

User-friendly interface for manual feature input through numeric fields and dropdowns

Real-time predictions with inverse transformation of target labels

Professional UI with proper styling and result visualization

Architecture Wireframe Diagram (Text-based)
text
[Raw Forest Cover Data (CSV)]
         │
         ▼
[Python Data Preprocessing (Jupyter)]
         │
         ├─ Outlier Detection & Treatment (IQR)
         ├─ Skewness Correction (Log Transform)
         ├─ Feature Engineering (6 New Features)
         ▼
[Cleaned & Enhanced Dataset]
         │
         ▼
[Feature Selection (Random Forest)]
         │
         ▼
[Multiple Model Training]
         ├─ Random Forest
         ├─ Decision Tree
         ├─ Logistic Regression
         ├─ K-Nearest Neighbors
         ├─ XGBoost
         │
         ▼
[Model Evaluation & Comparison]
         │
         ▼
[Hyperparameter Tuning (RandomizedSearchCV)]
         │
         ▼
[Best Model Selection & Serialization]
         │
         ▼
[Streamlit Web Application]
         ├─ Interactive User Inputs
         ├─ Real-time Predictions
         ├─ Forest Type Visualization
         ▼
[End-user Predictions & Insights]
Technical Implementation
Data Preprocessing & Cleaning
✅ Missing Values: Comprehensive analysis and treatment

✅ Outlier Detection: IQR (Interquartile Range) method for outlier identification

✅ Outlier Treatment: Capping using lower and upper bounds

✅ Skewness Treatment: Log transformation for highly skewed numerical features

✅ Data Validation: Duplicate removal and data integrity checks

Feature Engineering
Created 6 new derived features for improved model performance:

Hydrology_Distance_Ratio = Vertical_Distance_To_Hydrology / (Horizontal_Distance_To_Hydrology + 1)

Road_to_Hydrology_Ratio = Horizontal_Distance_To_Roadways / (Horizontal_Distance_To_Hydrology + 1)

Fire_to_Hydrology_Ratio = Horizontal_Distance_To_Fire_Points / (Horizontal_Distance_To_Hydrology + 1)

Hillshade_Morning_Noon_Diff = Hillshade_9am - Hillshade_Noon

Hillshade_Noon_Evening_Diff = Hillshade_Noon - Hillshade_3pm

Hillshade_Daily_Variation = abs(Hillshade_9am - Hillshade_3pm)

Feature Selection
Method: Random Forest Feature Importance

Top 20 Features selected from original 54+6 features

Criteria: Highest importance scores for optimal model performance

Model Development
Algorithms Implemented:

Random Forest Classifier (n_estimators=100, random_state=42)

Decision Tree Classifier (random_state=42)

Logistic Regression (max_iter=1000, random_state=42)

K-Nearest Neighbors (n_neighbors=5)

XGBoost Classifier (random_state=42)

Model Evaluation
Metrics Used:

Accuracy Score

Precision (Weighted)

Recall (Weighted)

Confusion Matrix

Classification Report

Hyperparameter Tuning
Best Model: K-Nearest Neighbors

Technique: RandomizedSearchCV with 5-fold cross-validation

Iterations: 50 parameter combinations

Parameters Tuned:

n_neighbors: 1 to 30

weights: ['uniform', 'distance']

algorithm: ['auto', 'ball_tree', 'kd_tree', 'brute']

leaf_size: [10, 20, 30, 40, 50]

p: [1, 2] (Distance metric)

Model Deployment
Framework: Streamlit Web Application

Model Serialization: Joblib for model, Pickle for label encoder

Input Handling: 20 feature inputs through interactive UI

Output: Real-time predictions with inverse label transformation

Dataset Information
Source: UCI Machine Learning Repository - Forest Cover Type

Samples: 581,012 observations

Features: 54 columns (including 6 derived features)

Target: 7 forest cover types

Target Classes:
Spruce/Fir 🌲

Lodgepole Pine 🪵

Ponderosa Pine 🌳

Cottonwood/Willow 🍃

Aspen 🍂

Douglas-fir 🎄

Krummholz 🌿

Model Performance Comparison
Model	Accuracy	Precision	Recall
Random Forest	0.8550	0.8570	0.8550
Decision Tree	0.8120	0.8150	0.8120
Logistic Regression	0.7230	0.7300	0.7230
K-Nearest Neighbors	0.8010	0.8050	0.8010
XGBoost	0.8480	0.8500	0.8480
Installation & Usage
Prerequisites
Python 3.8+

Jupyter Notebook

Streamlit

Dependencies
bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn streamlit joblib jupyter
Run the Application
bash
streamlit run app.py  
Technologies Used
Programming: Python 3.8+

Data Processing: Pandas, NumPy

Machine Learning: Scikit-learn, XGBoost

Visualization: Matplotlib, Seaborn

Web Framework: Streamlit

Model Serialization: Joblib, Pickle 

Author
Mukul - Complete implementation and deployment

Summary
The project flows from comprehensive data preprocessing and feature engineering, through multiple model training and evaluation, to optimized hyperparameter tuning and interactive web deployment. Each stage is modular, fully documented, and reproducible—enabling transparent, scalable, and impactful forest cover prediction for environmental and research applications.
