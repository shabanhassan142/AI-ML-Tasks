# AI-ML-Tasks

## Task1:
The objective of this task is to perform exploratory data analysis (EDA) and visualization on the classic Iris dataset. The goal is to understand the distribution, relationships, and characteristics of the features and classes within the dataset using various plots and summary statistics.

## Dataset Used
- **Dataset:** [Iris Dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set)
- **Source:** Built-in with Seaborn (`sns.load_dataset("iris")`)
- **Description:**  
  The Iris dataset contains 150 samples of iris flowers, with 4 features (sepal length, sepal width, petal length, petal width) and a target variable (species: setosa, versicolor, virginica).

---

## Models Applied
  This task focuses on data exploration and visualization using:
  - Pandas (for data inspection)
  - Seaborn and Matplotlib (for plotting)
---

## Key Results and Findings

### 1. **Scatter Plot: Sepal Length vs Petal Length**
- **What it shows:**  
  The relationship between sepal length and petal length, colored by species.
- **Finding:**  
  - Setosa is clearly separated from the other two species.
  - Versicolor and virginica show some overlap but are generally distinguishable.

### 2. **Histograms of Features**
- **What it shows:**  
  The distribution of each numerical feature (sepal length, sepal width, petal length, petal width).
- **Finding:**  
  - Features like petal length and petal width show clear separation between species.
  - Sepal width is more normally distributed, while sepal length is slightly skewed.

### 3. **Box Plot of Iris Features**
- **What it shows:**  
  The spread and outliers for each feature.
- **Finding:**  
  - Petal length and petal width have the largest spread.
  - Sepal width shows some outliers.

### 4. **Box Plot: Sepal Length by Species**
- **What it shows:**  
  The distribution of sepal length for each species.
- **Finding:**  
  - Setosa has the smallest sepal length, virginica the largest, and versicolor is intermediate.
  - There is little overlap between setosa and the other species, but some overlap between versicolor and virginica.

---

## Conclusion

- The Iris dataset is well-structured and shows clear differences between species, especially in petal measurements.
- Visualizations confirm that setosa is easily separable from the other species, while versicolor and virginica are more similar but still distinguishable.
- This EDA provides a strong foundation for further analysis or machine learning tasks, such as classification.

## Task2: Stock Price Prediction with Linear Regression
The objective of this task is to predict the next day's closing price of Apple (AAPL) stock using historical stock data and a linear regression model. The workflow includes data fetching, feature engineering, model training, evaluation, and visualization of results.


### Dataset Used

- **Source:** Yahoo Finance (via the `yfinance` Python library)
- **Date Range:** 2020-01-01 to 2023-01-01
- **Features Used:** Open, High, Low, Volume
- **Target:** Next day's Close price

---

### Models Applied

- **Linear Regression**
  - Used to model the relationship between the selected features and the next day's closing price.

---

### Key Results and Findings

- **Root Mean Squared Error (RMSE):** 3.37  
  This means that, on average, the model's predictions are about $3.37 away from the actual closing price.

- **R-squared (R²) Score:** 0.8967  
  This indicates that about 89.67% of the variance in the closing price can be explained by the model using the selected features.

- **Visualization:**  
  ![Actual vs Predicted Closing Prices] 
  The plot shows the actual vs. predicted closing prices for the test set. The lines are closely aligned, indicating good model performance.

---

### Interpretation

- The linear regression model performs well, as shown by the high R² score and low RMSE.
- The model captures the general trend of the stock price, but may not predict sudden spikes or drops perfectly (as is typical for stock data).
- This approach is a good baseline for stock price prediction, but more advanced models or additional features could further improve accuracy.

---

## Task3:  Heart Disease Prediction with Decision Tree

Predict the risk of heart disease in patients using health data and a Decision Tree classifier. The workflow includes data loading, cleaning, exploratory data analysis, model training, evaluation, and feature importance analysis.

---

### Dataset Used

- **Source:** [UCI Heart Disease Dataset](https://storage.googleapis.com/download.tensorflow.org/data/heart.csv)
- **Features:** Various patient health metrics (age, sex, cholesterol, etc.)
- **Target:** Presence of heart disease (1 = Disease, 0 = No Disease)

---

### Models Applied

- **Decision Tree Classifier**
  - Used to classify patients as at risk or not at risk of heart disease based on their health data.

---

### Key Results and Findings

- **Distribution of Target Variable:**
  ![Distribution of Heart Disease](bar plot)
  - The dataset is imbalanced, with more patients not having heart disease (target=0) than having it (target=1).

- **Correlation Matrix:**
  ![Correlation Matrix]
  - Shows relationships between features and the target variable. Some features are more strongly correlated with heart disease.

- **Confusion Matrix:**
  ![Confusion Matrix]
  - **Accuracy:** 73.77%
  - True Negatives (No Disease predicted correctly): 37
  - True Positives (Disease predicted correctly): 8
  - False Positives (No Disease predicted as Disease): 7
  - False Negatives (Disease predicted as No Disease): 9

- **ROC Curve & AUC:**
  ![ROC Curve]
  - **AUC Score:** 0.66
  - The ROC curve shows the trade-off between true positive and false positive rates. An AUC of 0.66 indicates moderate model performance.

---

### Interpretation

- The Decision Tree model achieves a moderate accuracy and AUC, indicating it can distinguish between patients with and without heart disease to a reasonable extent.
- The class imbalance may affect model performance; consider using resampling techniques for improvement.
- The correlation matrix and feature importance analysis help identify which health factors are most related to heart disease in this dataset.
- This is a simple model and should not be used for real medical decisions. Always consult a healthcare professional for actual diagnosis or treatment.

---
## Task 4: General Health Query Chatbot (Hugging Face LLM)

### Task Objective

Build a simple chatbot that can answer general health-related questions using a large language model (LLM) via the Hugging Face Inference API. The chatbot uses prompt engineering to provide friendly, safe, and responsible responses, and includes a safety filter to avoid giving harmful medical advice.

---

### Dataset / Model Used

- **Model:** [HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)
- **API:** Hugging Face Inference API
- **Data:** User-provided health-related questions (no training data required)

---

### Methods Applied

- **Prompt Engineering:**
  - The chatbot is instructed to act as a helpful, friendly, and responsible medical assistant.
  - It avoids giving specific medical advice, diagnoses, or recommendations for treatments, medications, or dosages.
  - Always encourages users to consult a healthcare professional for personal or urgent issues.
- **Safety Filter:**
  - Blocks queries containing unsafe or potentially harmful keywords (e.g., "prescribe", "overdose", "suicide").
- **API Integration:**
  - Uses the Hugging Face Inference API to send user queries to the LLM and return responses.

---

### Key Results and Findings

- The chatbot can answer a wide range of general health questions in a friendly and responsible manner.
- Example queries:
  - "What causes a sore throat?"
  - "Is paracetamol safe for children?"
  - "How can I improve my sleep quality?"
- The safety filter successfully blocks questions that could lead to harmful advice, ensuring responsible use.
- The chatbot always reminds users to consult a healthcare professional for personal or urgent medical concerns.

---

### Interpretation

- This chatbot demonstrates how LLMs can be used for general health information and education, but not for diagnosis or treatment.
- The safety filter and prompt engineering are essential for responsible AI use in sensitive domains like healthcare.
- The approach can be extended to other domains or improved with more advanced filtering and user experience features.

---

### Requirements

- Python 3.x
- requests
- A free Hugging Face account and access token (get one at https://huggingface.co/settings/tokens)

Install requirements with:
```bash
pip install requests
```

---

### How to Run

1. Open the script or notebook in your Python environment.
2. Run the code. When prompted, enter your Hugging Face access token (starts with `hf_...`).
3. Type your health-related question at the prompt. Type `exit` to quit.
4. Review the chatbot's responses and ensure you consult a healthcare professional for any personal or urgent issues. 
