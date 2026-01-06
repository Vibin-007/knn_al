# 🛍️ Social Network Ads Predictor (KNN)

A Streamlit application that predicts whether a user will purchase a product based on social network advertisement data using K-Nearest Neighbors (KNN).

## 📊 Features

- **Purchase Prediction**: Classifies users as "Purchased" or "Not Purchased".
- **Dynamic K Value**: Allows users to adjust the 'K' (neighbors) value to observe performance changes.
- **Decision Boundary**: Visualizes how the model separates classes in 2D space.
- **Performance Metrics**: Real-time calculation of Accuracy, Confusion Matrix, and Precision/Recall.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Vibin-007/knn_al.git
   cd knn_al
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

- `app.py`: Main application file containing the Streamlit interface and logic.
- `Social_Network_Ads.csv`: Dataset containing User ID, Gender, Age, EstimatedSalary, and Purchased status.
- `knn_analysis.ipynb`: Jupyter notebook for model development and analysis.
- `requirements.txt`: List of Python dependencies.

## 📈 Model Information

The model uses **K-Nearest Neighbors** to classify inputs based on:
- **Age**
- **Estimated Salary**
- **Gender**