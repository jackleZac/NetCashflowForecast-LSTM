# 💰 Daily Cash Flows Forecasting with LSTM

This project uses a **Long Short-Term Memory (LSTM)** neural network to predict future cashflow trends based on past financial data.  
It is designed to model sequential dependencies and capture complex temporal patterns in bank account data.

---

## 📘 Overview

The notebook focuses on analyzing and forecasting the cashflow of selected financial accounts — specifically **"Savings Bank Account 1"** and **"Cash"**.  
Using machine learning techniques, particularly **LSTM**, this project aims to predict future cash balances to assist with financial planning and analysis.

---

## ⚙️ Installation

Make sure you have Python 3.8+ installed, then install the dependencies:

```bash
pip install pandas numpy matplotlib tensorflow scikit-learn kagglehub
```

Alternatively, if you're running the notebook directly in Jupyter, these packages will be installed via:

```python
%pip install pandas numpy matplotlib tensorflow scikit-learn kagglehub
```

---

## 📂 Dataset

- The dataset contains multiple financial accounts.
- Only **"Savings Bank Account 1"** and **"Cash"** are used for prediction.
- Data is preprocessed with **RobustScaler** to handle outliers and ensure stable model training.

> **Note:** The dataset should be placed in the working directory or fetched automatically via KaggleHub if connected.

---

## 🧠 Model Architecture

The predictive model uses a **Sequential LSTM network** with:

- **LSTM layers** for temporal learning
- **Dropout** for regularization
- **Dense layers** for regression output
- **EarlyStopping** and **ModelCheckpoint** callbacks to prevent overfitting

The model predicts future cashflow values based on previous time steps.

---

## 📊 Evaluation

The model's performance is measured using:

- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- Visualization of actual vs. predicted cashflow trends using **Matplotlib**

---

## 🚀 Usage

1. Open the notebook:

```bash
jupyter notebook cashflow-prediction-with-lstm.ipynb
```

2. Run all cells sequentially.

3. View training progress, metrics, and cashflow forecast plots.

---

## 🧩 Technologies Used

- Python 3
- TensorFlow / Keras
- Scikit-learn
- Matplotlib
- NumPy / Pandas

---

## 📈 Results

The notebook demonstrates how deep learning can effectively model time series patterns in financial data. Predicted cashflow curves align closely with actual data, showing that LSTMs can learn underlying trends with appropriate scaling and tuning.

<img width="1121" height="583" alt="image" src="https://github.com/user-attachments/assets/e2aa5bbd-50cf-48d7-9265-f59e549e9ccb" />

---

## 🧑‍💻 Author

https://github.com/jackleZac

---

## 📜 License

This project is open-source under the **MIT License**. Feel free to use or modify it for educational or research purposes.
