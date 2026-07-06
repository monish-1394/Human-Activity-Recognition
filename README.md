# Human Activity Recognition using Smartphone Sensor Data

This project implements a machine learning pipeline for Human Activity Recognition (HAR) using the UCI HAR dataset. The goal is to classify human activities based on data collected from smartphone sensors. Feature reduction is performed using K-Means clustering, followed by model training and evaluation using Gaussian Naive Bayes and Logistic Regression.

## Dataset

**Source**: [UCI Machine Learning Repository - HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
**Features**: 561 sensor signals (accelerometer and gyroscope data)
**Labels**: 6 types of human activities (e.g., Walking, Sitting, Laying)

## Project Workflow

1. **Data Collection**  
   - Automatically downloaded and extracted from UCI's repository.

2. **Exploratory Data Analysis (EDA)**  
   - Basic data inspection, null checks, and feature analysis.

3. **Preprocessing**  
   - Label encoding of activity labels  
   - Feature scaling using `StandardScaler`

4. **Model Training (Baseline)**  
   - Gaussian Naive Bayes trained on full dataset  
   - Achieved ~73% accuracy

5. **Feature Reduction**  
   - K-Means clustering applied to feature space (transposed)  
   - 50 representative features selected from 561 total

6. **Optimized Training**  
   - Gaussian Naive Bayes retrained on selected features  
   - Improved accuracy to ~85%  
   - Reduced model training time
     
##  Technologies Used
- Python
- Scikit-learn
- NumPy, Pandas
- Matplotlib / Seaborn (optional for EDA)
- BeautifulSoup (for dataset parsing)
- DEAP (for optimization - optional future step)

## Result
|MODEL                      |Features Used|      Accuracy|
|---------------------------|--------------|--------------|
|Gaussian(baseline)         | 561          | ~73%         |
|GaussianNB + K-Means       | 50           |  ~85%        |


In the UCI HAR Dataset, the human activities identified (i.e., the labels in y_train.txt) are six predefined activities, each represented by a numeric code from 1 to 6. Here's the mapping:

|Label	 | Activity |
|----------|---------|
|1	     |  Walking   |
|2	     |  Walking Upstairs   |
|3      |  Walking Downstairs   |
|4	     |  Sitting            |
|5	     |  Standing            |
|6	     |  Laying               |

