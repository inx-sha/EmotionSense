import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import sys

def train_text_model(csv_path='data/text_labels.csv', model_out='models/text_clf.joblib'):
    print("📌 Loading CSV file:", csv_path)

    # LOAD CSV SAFELY
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print("❌ ERROR: Could not read CSV file:", e)
        sys.exit(1)

    # CHECK FOR REQUIRED COLUMNS
    if 'text' not in df.columns or 'emotion' not in df.columns:
        print("❌ ERROR: CSV must contain 'text' and 'emotion' columns.")
        print("Your CSV columns:", df.columns.tolist())
        sys.exit(1)

    print("✔ CSV loaded successfully.")
    print("Total rows:", len(df))

    # PREPARE DATA
    texts = df['text'].astype(str).tolist()
    labels = df['emotion'].astype(str).tolist()

    print("📌 First 5 labels:", labels[:5])
    
    # EMBEDDINGS

    print("\n📌 Loading embedding model (MiniLM)...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    print("📌 Encoding text...")
    embeddings = embed_model.encode(texts, show_progress_bar=True)

    # TRAIN/TEST SPLIT
    print("\n📌 Splitting dataset...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, 
            test_size=0.2, 
            random_state=42, 
            stratify=labels
        )
    except ValueError as e:
        print("❌ ERROR: Stratify failed — likely because one emotion has too few samples.")
        print("Details:", e)
        sys.exit(1)

    # TRAINING
    print("\n📌 Training classifier...")
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    # EVALUATION

    print("\n📌 Evaluation Report:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # SAVE MODEL
    print("\n📌 Saving model to:", model_out)
    joblib.dump({'clf': clf, 'embedder': 'all-MiniLM-L6-v2'}, model_out)

    print("🎉 Model saved successfully!")
    print("✅ You can now use this model in your Streamlit app.")


# RUN SCRIPT CORRECTLY

if __name__ == "__main__":
    train_text_model()