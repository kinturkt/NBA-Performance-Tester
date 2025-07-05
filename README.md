# NBA-Performance-Tester

An interactive **Streamlit** app to train, evaluate, and compare multiple machine learning models on the **NBA player stats dataset**. This tool helps visualize and benchmark model performance for classification tasks such as predicting player positions.

---

## 🚀 Features

- 🔍 Preview NBA dataset (stats, positions, performance)
- 🧪 Run individual ML models:
  - Decision Tree
  - k-Nearest Neighbors (kNN)
  - Naive Bayes
  - Support Vector Classifier (SVC)
  - Linear SVM
- 🛠️ Hyperparameter tuning (C, k, max depth)
- 📊 Visual comparison of Accuracy, Precision, Recall, F1-Score
- 🧾 Confusion matrices + classification reports
- ✅ 10-Fold Cross-Validation support

---

## 💡 How It Works

1. Launch the Streamlit app:
   ```bash
   streamlit run app.py

2. Preview the NBA dataset

3. Use the sidebar to tune hyperparameters

4. Click on any model to train & evaluate

5. See performance metrics and compare models visually

---

🧠 Technologies Used <br>

Python 3 <br>
Pandas – Data manipulation <br>
scikit-learn – ML algorithms & metrics <br>
Streamlit – Interactive front-end <br>
Matplotlib – Accuracy comparison chart <br>

--- 

📦 Setup & Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/kinturkt/NBA-performance-tester.git
   cd nba-performance-tester

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the app:
   ```bash
   streamlit run app.py

--- 
## 🌐 Live Demo

Try it live on Streamlit Cloud:  
👉 [Launch App](https://nba-model-performance-tester.streamlit.app/)

---

## Author <br>
Kintur Shah <br>
Built with curiosity and a love for clean ML workflows.
