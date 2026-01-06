# 🛍️ Social Network Ads Prediction (K-Nearest Neighbors)

This project implements a **K-Nearest Neighbors (KNN)** classifier to predict whether a user will purchase a product based on social network advertisement data.

## 🚀 Features

- **Purchase Prediction**: Classifies users as 'Purchased' or 'Not Purchased' based on Age and Estimated Salary.
- **Interactive UI**:
    - **Dynamic K Value**: Adjust the number of neighbors ($k$) and see results instantly.
    - **Decision Boundary**: Visualize how the model separates the two classes.
    - **Performance Metrics**: View Accuracy, Confusion Matrix, and Classification Report.

## 🛠️ Usage

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App**:
   ```bash
   python -m streamlit run app.py
   ```

## 📁 Project Structure

- `app.py`: Streamlit application file.
- `knn_analysis.ipynb`: Jupyter notebook for KNN analysis.
- `Social_Network_Ads.csv`: Dataset with User ID, Gender, Age, EstimatedSalary, and Purchased status.
- `requirements.txt`: Python package dependencies.

## 📂 Dataset

The project uses `Social_Network_Ads.csv`.

## 📦 Requirements

- streamlit
- pandas
- scikit-learn
- matplotlib
- seaborn