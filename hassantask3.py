import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import nltk
from nltk.corpus import stopwords
import string

# Download NLTK Data
nltk.download('stopwords')

# Sample Data Loading (This should be your dataset of essays and scores)
# For this example, we'll create some dummy data.
data = {
    'essay': [
        "This is a well-written essay with good grammar.",
        "This essay has some issues with spelling and structure.",
        "The content of this essay is excellent, but grammar could be improved.",
        "Poor grammar and lack of structure make this essay hard to read."
    ],
    'score': [8, 5, 7, 3]
}

df = pd.DataFrame(data)

# Step 1: Preprocessing Function
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Apply preprocessing to the essay column
df['cleaned_essay'] = df['essay'].apply(preprocess_text)

# Step 2: Feature Extraction using TF-IDF
# TF-IDF (Term Frequency-Inverse Document Frequency) is a common method for extracting features from text.
tfidf = TfidfVectorizer(max_features=100)  # Using 100 features for simplicity
X = tfidf.fit_transform(df['cleaned_essay']).toarray()
y = df['score']  # Scores

# Step 3: Splitting the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Building using Random Forest Regressor
rf = RandomForestRegressor(random_state=42)

# Create a pipeline with TF-IDF and the model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=100)),  # Same TF-IDF vectorizer
    ('model', rf)
])

# Step 5: Hyperparameter Tuning using Grid Search
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 10, 20]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(df['cleaned_essay'], df['score'])

# Step 6: Evaluate the Model
best_model = grid_search.best_estimator_

# Predicting on test data
y_pred = best_model.predict(X_test)

# Evaluation using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# If needed, the model can now be used to score new essays:
new_essay = ["This is a great essay with minor grammatical mistakes."]
new_essay_cleaned = [preprocess_text(essay) for essay in new_essay]
predicted_score = best_model.predict(new_essay_cleaned)
print(f"Predicted Score for New Essay: {predicted_score[0]}")