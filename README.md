# Student Grade Predictor

This is my first Machine Learning project. It predicts students' final grade (`G3`) and explores how the availability of previous grades affects prediction performance.

## Project Goal

The goal is to compare two useful prediction scenarios:

1. Early prediction without `G1` and `G2`, before previous period grades are available.
2. Prediction with `G1` and `G2`, when more academic information is available.

These experiments represent different use cases. The first can identify students earlier, while the second can make more informed predictions later in the school year.

## Dataset

The project uses the UCI Student Performance dataset. It contains demographic, social, school-related, and academic information about students. `G3` is the final grade and the prediction target, while `G1` and `G2` are grades from previous periods.

## Workflow

The notebook includes data quality checks, exploratory data analysis, and a Train/Validation/Test split. It uses one-hot encoding for categorical features and feature scaling for numerical features. I compare a baseline with Linear Regression and Ridge Regression, tune the Ridge regularization parameter, analyze prediction errors, and evaluate the selected model on the held-out Test set. The final preprocessing and model are combined in a scikit-learn `ColumnTransformer` and `Pipeline`, then saved with joblib.

## Experiments and Results

| Experiment | Model / Evaluation | MAE | RMSE |
| --- | --- | ---: | ---: |
| Early prediction (without G1/G2) | Baseline (Validation) | 3.79 | 4.90 |
| Early prediction (without G1/G2) | Linear Regression (Validation) | 3.53 | 4.52 |
| Early prediction (without G1/G2) | Ridge Regression, alpha = 50 (Validation) | 3.43 | 4.42 |
| With G1/G2 | Ridge Regression, alpha = 10 (Validation) | 1.49 | 2.40 |
| With G1/G2 | Ridge Regression (Final Test) | 1.45 | 2.02 |

## Key Findings

- Ridge Regression gave a modest improvement over ordinary Linear Regression for early prediction.
- Students with `G3 = 0` were responsible for several of the largest errors in the early model.
- Adding `G1` and `G2` substantially improved prediction accuracy.
- The experiments show the trade-off between predicting earlier and using more academic information.

## Final Model

The final model is Ridge Regression with `alpha = 10`. Categorical features are processed with `OneHotEncoder`, and numerical features are processed with `StandardScaler`. The preprocessing steps and model are combined in a scikit-learn `Pipeline`.

After model selection, the Pipeline is trained on the combined Train and Validation data and evaluated once on the held-out Test set. The trained Pipeline is stored in `student_grade_model.joblib`.

## Project Structure

```text
student-grade-predictor/
|-- Student_Grade_Predictor.ipynb
|-- student-mat.csv
|-- student_grade_model.joblib
|-- README.md
|-- requirements.txt
`-- .gitignore
```

## Running the Project

1. Clone the repository.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open `Student_Grade_Predictor.ipynb` in Jupyter.
4. Run the notebook cells in order.

The saved Pipeline can also be loaded without retraining:

```python
import joblib

model = joblib.load("student_grade_model.joblib")
```

## What I Practiced

- Exploratory Data Analysis
- Feature preprocessing
- Train/Validation/Test separation
- Linear and Ridge Regression
- Regularization and hyperparameter tuning
- Model evaluation with MAE and RMSE
- Error analysis
- scikit-learn Pipelines
- Model persistence with joblib
