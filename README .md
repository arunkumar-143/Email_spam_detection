# 📧 SMS Spam Classifier

A machine learning project that reads a text message and predicts whether it is **spam** or **ham** (not spam).
Built step by step in a Jupyter Notebook using Python and scikit-learn.

---

## 📌 Project Overview

Spam messages waste time and can be harmful (fake offers, scams). This project builds a simple text classifier
that learns from labelled messages and detects spam automatically.

**Pipeline:**

```
Load data → EDA → Text preprocessing → Bag of Words / TF-IDF → Train-test split → Train models → Evaluate → Predict
```

## 📂 Dataset

- **Name:** SMS Spam Collection (UCI Machine Learning Repository)
- **Size:** 5,572 messages (4,825 ham, 747 spam)
- **After removing 403 duplicates:** 5,169 messages
- **Columns used:** `v1` (label), `v2` (message text)

> The dataset is imbalanced (only about 13% spam), so accuracy alone is not enough. Precision, recall and F1 score are also used.

## 🛠️ Technologies Used

- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- NLTK (optional, for stopwords)
- Jupyter Notebook

## 🧹 Text Preprocessing

1. Convert text to lowercase
2. Remove punctuation and numbers
3. Tokenization (split into words)
4. Remove stopwords

## 🔢 Feature Extraction

- **Bag of Words** (`CountVectorizer`)
- **TF-IDF** (`TfidfVectorizer`)

The vectorizer is fitted on the training data only, to avoid data leakage.

## 🤖 Models

- Multinomial Naive Bayes
- Logistic Regression

## 📊 Results (20% test set)

| Model + Features | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Naive Bayes + BoW | 0.9826 | 0.9449 | 0.9160 | **0.9302** |
| Naive Bayes + TF-IDF | 0.9662 | 0.9898 | 0.7405 | 0.8472 |
| Logistic Regression + BoW | 0.9768 | 0.9908 | 0.8244 | 0.9000 |
| Logistic Regression + TF-IDF | 0.9594 | 0.9890 | 0.6870 | 0.8108 |

**Key findings**
- Best overall: **Naive Bayes + Bag of Words** (F1 = 0.93)
- Precision is high for all models, so real messages are rarely marked as spam
- TF-IDF models are very careful but miss more spam (lower recall)
- Top spam words: `txt`, `mobile`, `call`, `reply`, `claim`, `stop`, `free`

## 📁 Project Structure

```
spam-classifier/
│
├── Spam_Classifier.ipynb   # Complete step-by-step notebook
├── spam.csv                # Dataset
├── requirements.txt        # Python libraries
└── README.md
```

## ⚙️ How to Run

**1. Clone the repository or download the files**
```bash
git clone <your-repository-url>
cd spam-classifier
```

**2. (Optional) Create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux
```

**3. Install the libraries**
```bash
pip install -r requirements.txt
```

**4. Open the notebook**
```bash
jupyter notebook Spam_Classifier.ipynb
```

Then click **Run → Run All Cells**. Keep `spam.csv` in the same folder as the notebook.

## 🔍 Example Predictions

| Message | Prediction |
|---|---|
| "Congratulations! You have won a free prize. Call 0800 now to claim your reward" | SPAM |
| "Hey, are we meeting for lunch tomorrow?" | HAM |

## 🚀 Future Improvements

- Add stemming or lemmatization
- Use n-grams (`ngram_range=(1, 2)`) and keep numbers as features
- Improve recall with `class_weight='balanced'`
- Try SVM and Random Forest, and tune with GridSearchCV
- Save the model with `joblib` and build a Streamlit web app

## 👤 Author

**Gowtham K R**
BCA (Computer Science) student
GitHub: [gowthamkr-55](https://github.com/gowthamkr-55)

## 📄 License

This project is for learning and educational purposes.
