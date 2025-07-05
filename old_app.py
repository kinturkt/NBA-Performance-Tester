import streamlit as st
import pandas as pd
import subprocess
import os
import matplotlib.pyplot as plt
import re

# Initialize session state for results
if "results" not in st.session_state:
    st.session_state.results = {}

if "available_models" not in st.session_state:
    st.session_state.available_models = []

# Title
st.title("Model Performance Tester")

# Dataset Information
st.write("### About the NBA Stats Dataset")
st.write("This dataset contains statistics of NBA players, including points, assists, rebounds, shooting percentages, and player positions.")
st.write("The dataset is used to train machine learning models to predict player positions based on their stats.")

# Show dataset preview
st.write("### Dataset Preview:")
dataset_path = "nba_stats.csv"  # Default dataset path
df = pd.read_csv(dataset_path)
st.write(df.head())

st.write("### Run Individual Models:")

# Model options
models = {
    "Decision Tree": "decision_tree.py",
    "kNN Classifier": "knn.py",
    "Naive Bayes": "naive_bayes.py",
    "Support Vector Classifier (SVC)": "svc.py",
    "Linear SVM": "linear_svm.py"
}

# Hyperparameter tuning options
st.sidebar.write("## Hyperparameter Tuning")
hyperparams = {
    "Decision Tree": st.sidebar.slider("Maximum Depth (Decision Tree)", 1, 20, 6),
    "kNN Classifier": st.sidebar.slider("Number of Neighbors (kNN)", 1, 50, 5),
    "Naive Bayes": None,  # No hyperparameter needed
    "Support Vector Classifier (SVC)": st.sidebar.slider("Regularization Parameter C (SVC)", 0.1, 100.0, 1.0),
    "Linear SVM": st.sidebar.slider("Regularization Parameter C (Linear SVM)", 0.1, 10.0, 1.0)
}

for model_name, script in models.items():
    if st.button(f"Run {model_name}"):
        st.write(f"Running {model_name} with hyperparameters...")
        
        if os.path.exists(script):
            process = subprocess.Popen(["python", script, str(hyperparams[model_name])], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output, error = process.communicate()
            
            if error:
                st.error(f"Error running {model_name}: {error}")
            else:
                st.success(f"{model_name} completed successfully!")
                with st.expander(f"See {model_name} Output"):
                    st.text_area(f"Output of {model_name}", output, height=200)
                
                # Extract accuracy and save results
                try:
                    mean_accuracy_match = re.search(r"Mean CV Accuracy of .*?: ([0-9]*\.?[0-9]+)", output)
                    validation_accuracy_match = re.search(r"Validation Set Accuracy: ([0-9]*\.?[0-9]+)%", output)
                    precision_match = re.search(r"Precision: ([0-9]*\.?[0-9]+)%", output)
                    recall_match = re.search(r"Recall: ([0-9]*\.?[0-9]+)%", output)
                    f1_score_match = re.search(r"F1-Score: ([0-9]*\.?[0-9]+)%", output)
                    
                    accuracy_value = None
                    if mean_accuracy_match:
                        accuracy_value = float(mean_accuracy_match.group(1)) * 100  # Convert decimal to percentage
                    elif validation_accuracy_match:
                        accuracy_value = float(validation_accuracy_match.group(1))
                    
                    precision = float(precision_match.group(1)) if precision_match else None
                    recall = float(recall_match.group(1)) if recall_match else None
                    f1_score = float(f1_score_match.group(1)) if f1_score_match else None
                    
                    if accuracy_value is not None:
                        # Save accuracy and additional metrics
                        st.session_state.results[model_name] = {
                            "Accuracy": accuracy_value,
                            "Precision": precision,
                            "Recall": recall,
                            "F1-Score": f1_score
                        }
                        if model_name not in st.session_state.available_models:
                            st.session_state.available_models.append(model_name)
                        
                        result_df = pd.DataFrame({
                            "Model": [model_name],
                            "Accuracy": [accuracy_value],
                            "Precision": [precision],
                            "Recall": [recall],
                            "F1-Score": [f1_score]
                        })
                        results_file = "model_results.csv"
                        if os.path.exists(results_file):
                            result_df.to_csv(results_file, mode="a", header=False, index=False)
                        else:
                            result_df.to_csv(results_file, index=False)
                    else:
                        st.warning(f"Could not extract accuracy for {model_name}. Ensure model outputs accuracy in expected format.")
                except Exception as e:
                    st.warning(f"Could not extract accuracy for {model_name}: {e}")

# Compare Model Performances
st.write("### Compare Models")

if st.session_state.results:
    selected_models = st.multiselect(
        "Select models to compare:",
        options=st.session_state.available_models,
        key="selected_models"
    )
    
    if st.button("Compare Selected Models"):
        if selected_models:
            # Create a DataFrame for selected models
            comparison_data = {
                "Model": [],
                "Accuracy": [],
                "Precision": [],
                "Recall": [],
                "F1-Score": []
            }
            for model in selected_models:
                comparison_data["Model"].append(model)
                comparison_data["Accuracy"].append(st.session_state.results[model]["Accuracy"])
                comparison_data["Precision"].append(st.session_state.results[model]["Precision"])
                comparison_data["Recall"].append(st.session_state.results[model]["Recall"])
                comparison_data["F1-Score"].append(st.session_state.results[model]["F1-Score"])
            
            comparison_df = pd.DataFrame(comparison_data)
            st.write(comparison_df)

            # Plot comparison chart
            plt.figure(figsize=(10, 5))
            plt.bar(comparison_df["Model"], comparison_df["Accuracy"], color=["blue", "green", "red", "purple", "orange"][:len(comparison_df)])
            plt.xlabel("Models")
            plt.ylabel("Accuracy (%)")
            plt.title("Comparison of Model Accuracies")
            st.pyplot(plt)
        else:
            st.error("Please select at least one model to compare.")
else:
    st.error("No results found. Please run models first.")