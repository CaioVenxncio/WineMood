# WineMood - Wine Review Sentiment Classifier

![WineMood Dashboard](./screenshots/wine_dashboard.png)


**WineMood** is an intelligent system that analyzes textual wine reviews and automatically classifies the expressed sentiment as **positive**, **neutral**, or **negative**. Using Natural Language Processing (NLP) techniques combined with Deep Learning, the project transforms human language into useful insights for consumers and businesses.

## 📋 Overview

This classifier can be integrated into wine e-commerce platforms, such as Vinheria Agnello, providing customers with automatic sentiment analysis of reviews before purchasing.

### 🎯 Objectives

- Apply NLP and Deep Learning techniques to real-world data
- Build a model capable of understanding and classifying emotions in text
- Develop a user-friendly web interface for real-time testing
- Present a practical and applicable proof of concept for businesses

## 🧰 Technologies and Tools

| Category                 | Technology            |
|--------------------------|-----------------------|
| Programming Language     | Python                |
| Deep Learning Framework  | TensorFlow + Keras    |
| Data Manipulation        | Pandas, NumPy         |
| NLP & Preprocessing      | NLTK, Regex, Tokenizer|
| Web Interface            | Streamlit             |
| Dataset                  | Wine Reviews (Kaggle) |
| Training Platform        | Google Colab (Free GPU)|
| Visualization            | Matplotlib / Streamlit|
| Version Control          | GitHub                |

## 📦 Dataset

- **Name**: Wine Reviews Dataset
- **Source**: Kaggle (https://www.kaggle.com/datasets/zynicide/wine-reviews)
- **Records**: ~130,000 wine reviews
- **Used Columns**:
  - `description`: text of the wine review
  - `points`: wine score (from 80 to 100)
- **Derived Label**: Sentiment (positive, neutral, negative) based on score

## 🔧 Development Steps

### 1. Data Collection and Exploration
- Load dataset using pandas
- Check for null values, class distribution, and descriptive statistics

### 2. Label Creation
- Convert numerical scores into sentiment categories:
  - ≥ 90: Positive
  - 80–89: Neutral
  - < 80: Negative

### 3. Text Preprocessing
- Clean text (lowercase, punctuation and stopwords removal)
- Tokenize and standardize sequences to fixed length

### 4. Deep Learning Model Training
- **Model architecture**:
  - Embedding for text vectorization
  - LSTM for sequence understanding
  - Dense with softmax for classification
- Train/test split (80/20)
- Metrics: accuracy, loss

### 5. Evaluation
- Evaluate model on test set using `.evaluate()`
- Print accuracy and analyze incorrect predictions

### 6. Web Interface (Streamlit)
- Web interface where users input text reviews
- The model predicts and displays detected sentiment
- Extra features:
  - Bar chart showing sentiment percentages
  - Process multiple reviews at once
  - “About the Project” tab with explanations

## 🖼 Example Usage

**User Input**:
"Great complexity with floral aromas and a long finish."

**Predicted Output**:
Sentiment: Positive

## 🌍 Real-World Applications

- Automatically recommend wines based on customer sentiment
- Help wine sellers and producers understand public perception
- Highlight top-rated wines on e-commerce platforms

## 📈 Expected Results

- Accuracy above 80% with balanced training data
- Lightweight, responsive web interface
- Modular, reusable code for similar sentiment analysis tasks

## 🚀 Project Highlights

- Fully functional project with real-time interface
- Uses real-world data with practical applications
- Combines NLP, LSTM, and Web Deployment
- Ready for commercial platform integration

## 📝 Usage Instructions

### Local Execution

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Streamlit app: `streamlit run app.py`

### Google Colab

1. Open the notebook `colab_notebook.ipynb` in Google Colab
2. Run all cells to train the model and generate necessary files
3. Download generated files (model.h5, tokenizer.pickle) for local use

### Online Deployment

The app can be deployed on Streamlit Cloud for public access:

1. Create a Streamlit Cloud account
2. Connect your GitHub repository with the project files
3. Set the main file as app.py for deployment

## 👥 Contributions

Contributions are welcome! Feel free to open issues or submit pull requests with improvements.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
