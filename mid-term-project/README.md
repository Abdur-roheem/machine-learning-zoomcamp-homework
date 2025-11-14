# **📘 Term Deposit Subscription Prediction**

## **📌 Project Overview**

This project builds a machine-learning model to predict whether a bank customer will subscribe to a term deposit.
**The pipeline includes:**

> Data exploration & preprocessing

> Feature engineering

> Model training using Logistic Regression and XGBoost

> Model evaluation using ROC–AUC

> Saving and deploying the final model

> Dockerized prediction service

The dataset is from direct marketing campaigns conducted by a Portuguese banking institution.

The solution is designed to support automated bank marketing decisions, enabling institutions to identify high-probability subscribers and optimise marketing campaigns.

**A complete workflow is included:**

> Data exploration and preprocessing

> Model training using multiple algorithms

> Model selection based on performance metrics (e.g., ROC AUC)

> Exporting the final model into a portable .bin file

> Loading the model for prediction through a Python script

> Optional Docker support for deployment

The project uses the Bank Marketing Dataset (bank-full.csv).

**2. Files in the Repository**
+  _.python-version_     # Python version specification for environment managers
+ _Dockerfile_           # Docker image configuration for deployment
+ _README.md_            # Project documentation (this file)
+ _bank-full.csv_        # Dataset used for training and evaluation
+ _bank-model.bin_       # Serialized DictVectorizer + trained ML model
+ _notebook.ipynb_       # Full workflow: EDA, preprocessing, model testing, saving
+ _predict.py_           # Script to load model.bin and predict for new customer data
+ _predict_old.py_       # Earlier version of prediction script (kept for reference)
+ _pyproject.toml_       # Project dependencies and build configuration
+ _test.py_              # Basic test script to validate model loading/prediction
+ _train.py_             # End-to-end training script for model construction
+ _uv.lock_              # Lock file for deterministic package resolution

**4. How to Run the Project**
  A. Running Locally (Without Docker)
        1. Clone the repository
           + git clone https://github.comAbdur-roheem/machine-learning-zoomcamp-homework/edit/main/mid-term-project/prediction.git
           + cd term-deposit-prediction

        2. Install dependencies
            - If you're using uv:
            
                `uv sync`
                
                Or using pip:

                `pip install -r requirements.txt`

_(If requirements.txt is not generated, dependencies are inside pyproject.toml.)_

        3. Train the model (optional)

            - This will regenerate bank-model.bin.

              `python train.py`

        4. Make a prediction
            `python predict.py` --input '{"age": 35, "job": "technician", "balance": 1500, "loan": "no"}'

    B. Running via Docker
      1. Build the Docker image
        `docker build -t term-deposit-predict .`

      2. Run the container
          `docker run -p 9696:9696 term-deposit-predict`

      3. Send a prediction request
          curl -X POST \
            -H "Content-Type: application/json" \
            -d '{"age": 40, "job": "admin", "balance": 2000, "loan": "no"}' \
            http://localhost:9696/predict

      5. Model Development Workflow

All data science steps are documented in notebook.ipynb, including:

- Data collection and initial checks

- Missing-value treatment

- Exploratory Data Analysis (EDA)

- Feature preprocessing

- Training multiple models (e.g., Logistic Regression, XGBoost)

- Comparing performance using ROC AUC and other metrics

- Selection of the best model

- Saving the final model and DictVectorizer into bank-model.bin

This ensures the notebook serves as a full methodological record suitable for academic review.

      6. Deployment Architecture

**The project uses:**

> DictVectorizer for feature encoding

> XGBoost for prediction

> Model serialised using pickle into bank-model.bin

> A lightweight prediction script for input scoring

> Optional Docker container for consistent deployment

This makes the system portable, reproducible, and ready for production.

7. Author / Credits

Project developed as part of an applied machine-learning workflow for predicting customer subscription behavior in banking marketing campaigns.
