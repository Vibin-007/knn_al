import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Purchase Predictor", page_icon="🛒", layout="wide")

# Title and description
st.title("🛒 Customer Purchase Prediction")
st.markdown("### Predict whether a customer will make a purchase based on Age and Estimated Salary")

# Load and train model
@st.cache_resource
def load_model():
    # Load data
    df = pd.read_csv("Social_Network_Ads.csv")
    x = df[['Age', 'EstimatedSalary']].values
    y = df['Purchased'].values
    
    # Train model
    model = KNeighborsClassifier(n_neighbors=100)
    model.fit(x, y)
    
    return model, x, y, df

try:
    model, x, y, df = load_model()
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Make a Prediction")
        
        # Input fields
        age = st.slider("Age", min_value=18, max_value=65, value=30, step=1)
        salary = st.slider("Estimated Salary ($)", min_value=15000, max_value=150000, value=75000, step=1000)
        
        # Predict button
        if st.button("🔮 Predict Purchase Behavior", type="primary"):
            newdata = [[age, salary]]
            prediction = model.predict(newdata)[0]
            
            # Display prediction
            st.markdown("---")
            if prediction == 1:
                st.success("✅ **Prediction: WILL PURCHASE**")
                st.balloons()
            else:
                st.error("❌ **Prediction: WILL NOT PURCHASE**")
            
            # Show confidence (distance to nearest neighbors)
            distances, indices = model.kneighbors(newdata)
            avg_distance = np.mean(distances)
            
            st.info(f"Average distance to {model.n_neighbors} nearest neighbors: {avg_distance:.2f}")
    
    with col2:
        st.subheader("📈 Data Visualization")
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot existing data
        colors = ['red' if val == 0 else 'green' for val in y]
        ax.scatter(x[:, 0], x[:, 1], c=colors, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Plot prediction point if button was clicked
        if 'age' in locals() and st.session_state.get('show_prediction', False):
            ax.scatter([age], [salary], c='blue', s=200, marker='*', 
                      edgecolors='black', linewidth=2, label='Your Input', zorder=5)
        
        ax.set_xlabel("Age", fontsize=12, fontweight='bold')
        ax.set_ylabel("Estimated Salary ($)", fontsize=12, fontweight='bold')
        ax.set_title("Customer Purchase Distribution", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', edgecolor='black', label='No Purchase'),
            Patch(facecolor='green', edgecolor='black', label='Purchase')
        ]
        ax.legend(handles=legend_elements, loc='upper left')
        
        st.pyplot(fig)
    
    # Display dataset statistics
    st.markdown("---")
    st.subheader("📋 Dataset Statistics")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.metric("Total Customers", len(df))
    
    with col4:
        purchased = df['Purchased'].sum()
        st.metric("Purchased", f"{purchased} ({purchased/len(df)*100:.1f}%)")
    
    with col5:
        not_purchased = len(df) - purchased
        st.metric("Not Purchased", f"{not_purchased} ({not_purchased/len(df)*100:.1f}%)")
    
    # Show sample data
    with st.expander("🔍 View Sample Data"):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Model information
    with st.expander("ℹ️ Model Information"):
        st.write(f"**Algorithm:** K-Nearest Neighbors (KNN)")
        st.write(f"**Number of Neighbors (k):** {model.n_neighbors}")
        st.write(f"**Training Samples:** {len(x)}")
        st.write(f"**Features:** Age, Estimated Salary")

except FileNotFoundError:
    st.error("⚠️ Error: 'Social_Network_Ads.csv' file not found. Please ensure the CSV file is in the same directory as this script.")
except Exception as e:
    st.error(f"⚠️ An error occurred: {str(e)}")

# Footer
st.markdown("---")
