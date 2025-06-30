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

- ## Task2: Stock Price Prediction with Linear Regression
- 
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
  ![Actual vs Predicted Closing Prices](attachment:image1.png)  
  The plot shows the actual vs. predicted closing prices for the test set. The lines are closely aligned, indicating good model performance.

---

### Interpretation

- The linear regression model performs well, as shown by the high R² score and low RMSE.
- The model captures the general trend of the stock price, but may not predict sudden spikes or drops perfectly (as is typical for stock data).
- This approach is a good baseline for stock price prediction, but more advanced models or additional features could further improve accuracy.

---

## Task3: 
