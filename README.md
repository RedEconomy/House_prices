# Melbourne Housing Price Prediction

## 📌 Project Overview

This project predicts house prices in Melbourne, Australia using machine learning models. Various regression techniques, including **Decision Trees**, **Random Forests**, and **Linear Regression**, are implemented and compared based on **Mean Absolute Error (MAE)**.
_Disclaimer: The purpose of this project is not getting the lowest MAE possible using all methods or variables, but showcase the diverse skillsets used. Decreasing the MAE further would require multiple other variables in the dataset and/or other techniques._
## 🗂 Dataset

The dataset used is **melb\_data.csv**, containing information on Melbourne housing prices. Key features (used) include:

- `Rooms`: Number of rooms in the house
- `Bathroom`: Number of bathrooms
- `Landsize`: Land area in square meters
- `Distance`: Distance from the Melbourne Central Business District (CBD)
- `Price`: The target variable (house price)

## 🏗 Methodology

### **1️⃣ Data Preprocessing**

- Removed entries with missing `Price` values
- Imputed missing feature values using **median imputation**
- Removed rows where `Landsize` is 0 (to compute price per square meter)
- Created a new feature: `Price_per_m2`
- Standardized features **only** for Linear Regression

### **2️⃣ Model Selection & Training**

- **Decision Tree** with GridSearchCV 
- **Random Forest** with GridSearchCV
- **Linear Regression** (Scaled)
- **Random Forest with New Features (Price\_per\_m2)**
_Explanation for Price per m2: Captures the perceived value of the land. Large property does not equal high value. Acts as proxy for environmental worth_ 
  
### **3️⃣ Performance Evaluation**

- Models are evaluated using **Mean Absolute Error (MAE)**
- A results table is printed, ranking models by lowest MAE

## 📊 Results

| Model                        | MAE (Lower is better)      |
| ---------------------------- | ---------------------      |
| Random Forest                | **XXXX.XX**               |
| New Features RF              | **XXXX.XX**               |
| Decision Tree                | **XXXX.XX**               |
| Linear Regression            | **XXXX.XX**               |

## 🛠 Dependencies

To run this project, install the following packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## 🚀 How to Run

1. Place `melb_data.csv` in the same directory as the script
2. Run the Python script:
   ```bash
   python melbourne_housing.py
   ```
3. The script will print model performance and generate the following graphs:
   - Feature Importance 
   - Correlation Heatmap

## 📌 Further Improvements

- Use additional features for improved accuracy
- Try advanced models (e.g., XGBoost, Gradient Boosting)
- Implement cross-validation for better generalization

---

**Author**: *Jack Hou*
📅 **Last Updated**: *26/02/2025*



