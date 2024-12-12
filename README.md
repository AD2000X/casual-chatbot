# Intent Classification Chatbot

## Overview
This project implements an intent classification chatbot using various machine learning models. The system processes user input, classifies intents using ELMo embeddings, and selects the most appropriate response based on the predicted intent.

## Project Structure
The project is structured as a single Python file (`chatbot.py`), which includes:
- Data preprocessing and analysis
- Model training and evaluation
- Chatbot implementation
- Flask API setup (commented out for future work)

## Features
- Comprehensive EDA with visualizations
- Text preprocessing and cleaning
- ELMo embeddings for text representation
- Multiple machine learning models:
  - Support Vector Machine (SVM)
  - Random Forest
  - Gradient Boosting
  - Logistic Regression
- Model performance evaluation
- Interactive chatbot interface

## Dependencies
All required packages are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

## Data Requirements
The system requires an input file named `dialogs.csv` with the following columns:
- **input**: User messages
- **output**: Bot responses
- **intent**: Message intents

## Usage

### Data Analysis and Model Training
Run the following command to start data analysis and model training:

```bash
python chatbot.py
```

The script will:
- Perform EDA with visualizations
- Preprocess text data
- Train multiple models
- Save the best performing model
- Implement chatbot functionality

### Model Training
The system trains and evaluates four different models:
- **Support Vector Machine (SVM)** (with grid search)
- **Random Forest**
- **Gradient Boosting**
- **Logistic Regression**

Each model is evaluated using the following metrics:
- Accuracy
- F1 Score
- Precision
- Recall
- Confusion Matrix

## Output Files
- Trained model: `best_logistic_regression_model.joblib`
- Various visualization plots
- Performance metrics

## Future Work
- Implementation of a Flask web interface
- Model deployment
- Response generation improvements
- Additional NLP features
