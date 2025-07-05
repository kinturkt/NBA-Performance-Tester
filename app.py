import streamlit as st
import pandas as pd
import subprocess
import os
import matplotlib.pyplot as plt
import re

if "results" not in st.session_state:
    st.session_state.results = {}
if "available_models" not in st.session_state:
    st.session_state.available_models = []

# --- Title ---
st.title("Model Performance Tester")

# --- Load NBA Dataset ---
dataset_path = "nba_stats.csv"
df = pd.read_csv(dataset_path)
target_column = "Pos"

st.write("### About the NBA Stats Dataset")
st.write("This dataset contains statistics of NBA players, including points, assists, rebounds, shooting percentages, and player positions.")
st.write("It is used to train machine learning models to predict player positions.")

st.write("### Dataset Preview")
st.write(df.head())

st.write("### Run Individual Models")

models = {
    "Decision Tree": "decision_tree.py",
    "kNN Classifier": "knn.py",
    "Naive Bayes": "naive_bayes.py",
    "Support Vector Classifier (SVC)": "svc.py",
    "Linear SVM": "linear_svm.py"
}

# --- Hyperparameters Sidebars ---
st.sidebar.write("## Hyperparameter Tuning")
hyperparams = {
    "Decision Tree": st.sidebar.slider("Max Depth (Decision Tree)", 1, 20, 6),
    "kNN Classifier": st.sidebar.slider("k (kNN)", 1, 50, 5),
    "Naive Bayes": None,
    "Support Vector Classifier (SVC)": st.sidebar.slider("C (SVC)", 0.1, 100.0, 1.0),
    "Linear SVM": st.sidebar.slider("C (Linear SVM)", 0.1, 10.0, 1.0)
}

for model_name, script in models.items():
    if st.button(f"Run {model_name}"):
        st.write(f"Running {model_name}...")

        if not os.path.exists(script):
            st.error(f"Script not found: {script}")
            continue

        cmd = ["python", script]
        if hyperparams[model_name] is not None:
            cmd.append(str(hyperparams[model_name]))
        cmd.append(dataset_path)
        cmd.append(target_column)

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output, error = process.communicate()

        if error:
            st.error(f"Error running {model_name}:\n{error}")
        else:
            st.success(f"{model_name} completed successfully!")
            with st.expander(f"See {model_name} Output"):
                st.text_area("Output", output, height=250)

            try:
                acc_match = re.search(r"Validation Set Accuracy: ([0-9]*\.?[0-9]+)%", output)
                precision_match = re.search(r"Precision: ([0-9]*\.?[0-9]+)%", output)
                recall_match = re.search(r"Recall: ([0-9]*\.?[0-9]+)%", output)
                f1_match = re.search(r"F1-Score: ([0-9]*\.?[0-9]+)%", output)

                acc = float(acc_match.group(1)) if acc_match else None
                precision = float(precision_match.group(1)) if precision_match else None
                recall = float(recall_match.group(1)) if recall_match else None
                f1 = float(f1_match.group(1)) if f1_match else None

                if acc is not None:
                    st.session_state.results[model_name] = {
                        "Accuracy": acc,
                        "Precision": precision,
                        "Recall": recall,
                        "F1-Score": f1
                    }
                    if model_name not in st.session_state.available_models:
                        st.session_state.available_models.append(model_name)

                    st.metric("Accuracy", f"{acc:.2f}%")
                    st.metric("Precision", f"{precision:.2f}%")
                    st.metric("Recall", f"{recall:.2f}%")
                    st.metric("F1-Score", f"{f1:.2f}%")

            except Exception as e:
                st.warning(f"Could not extract metrics: {e}")

# --- Compare the Models ---
st.write("### Compare Models")
if st.session_state.results:
    selected = st.multiselect("Select models to compare:", st.session_state.available_models)

    if st.button("Compare Selected Models"):
        if selected:
            comp_data = {
                "Model": [],
                "Accuracy": [],
                "Precision": [],
                "Recall": [],
                "F1-Score": []
            }
            for model in selected:
                res = st.session_state.results[model]
                comp_data["Model"].append(model)
                comp_data["Accuracy"].append(res["Accuracy"])
                comp_data["Precision"].append(res["Precision"])
                comp_data["Recall"].append(res["Recall"])
                comp_data["F1-Score"].append(res["F1-Score"])

            comp_df = pd.DataFrame(comp_data)
            st.write(comp_df)

            # --- Plot ---
            plt.figure(figsize=(10, 5))
            bars = plt.bar(comp_df["Model"], comp_df["Accuracy"], color=['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd'])

            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f"{height:.1f}%", ha='center', va='bottom', fontsize=10)

            plt.xlabel("Model")
            plt.ylabel("Accuracy (%)")
            plt.title("Model Accuracy Comparison")
            plt.ylim(0, 110)
            st.pyplot(plt)

        else:
            st.warning("Please select at least one model.")
else:
    st.info("No models have been run yet.")
