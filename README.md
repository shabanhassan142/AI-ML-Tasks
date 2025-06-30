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

- ## Task2
- 
