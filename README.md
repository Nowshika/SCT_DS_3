# 📊 Bank Marketing Campaign Analysis & Customer Subscription Prediction

## 📌 Objective
Predict whether a customer will subscribe to a term deposit based on features from a Portuguese banking institution’s direct marketing campaigns.

## 📁 Dataset
- Source: [UCI Machine Learning Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- Files used: `bank-full.csv` (subsampled for performance)

## 🔧 Key Features
- age, job, marital status, education
- campaign, previous outcome, call duration
- custom features: `age_duration_interaction`, `total_contacts`, binned age & duration

## 🧪 Models Trained
- Decision Tree (visualized, pruned)
- Random Forest 🌟 *(Best Accuracy: ~90%)*
- Gradient Boosting
- Logistic Regression
- Support Vector Machine (Linear SVC)
- K-Nearest Neighbors
- Gaussian Naive Bayes

## 📈 Evaluation Metrics
- Accuracy
- Classification Report
- Confusion Matrix
- ROC-AUC Curve
- 5-Fold Cross-Validation

## 📊 Visualizations
- Confusion Matrices with custom colormap
- ROC-AUC Curves
- Feature Importance for Tree Models
- Pruned Decision Tree Visualization with Rule Export

## 💡 Business Insights
- Call `duration` and previous outcomes (`poutcome`) are the most influential features.
- Optimize campaign costs by skipping long calls with low success probability.
- A/B test top features and focus on high-likelihood segments for targeting.

## 📎 Tech Stack
- Python, Pandas, Scikit-learn, Seaborn, Matplotlib

## ✅ Status
Analysis complete ✅  
Model ready for deployment or dashboard integration 🔍📊

---

### 📚 To Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python bank_marketing_analysis.py
