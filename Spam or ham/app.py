# app.py

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report, accuracy_score
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Function to load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
    df.rename(columns={"v1": "Category", "v2": "Message"}, inplace=True)
    df.drop(columns={'Unnamed: 2','Unnamed: 3','Unnamed: 4'}, inplace=True)
    df['Spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
    return df

# Function to visualize data
def visualize_data(df):
    st.subheader("Dataset Overview")
    st.write(df.head())

    st.subheader("Dataset Info")
    st.write(df.info())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Category Distribution")
    spread = df['Category'].value_counts()
    fig1, ax1 = plt.subplots()
    spread.plot(kind='pie', autopct='%1.2f%%', cmap='Set1', ax=ax1)
    plt.title('Distribution of Spam vs Ham')
    st.pyplot(fig1)

    st.subheader("Most Used Words in Spam Messages")
    df_spam = df[df['Category'] == 'spam'].copy()
    comment_words = ''
    stopwords = set(STOPWORDS)
    for val in df_spam.Message:
        val = str(val)
        tokens = val.split()
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        comment_words += " ".join(tokens) + " "
    wordcloud = WordCloud(width=800, height=400,
                          background_color='white',
                          stopwords=stopwords,
                          min_font_size=10,
                          max_words=1000,
                          colormap='gist_heat_r').generate(comment_words)
    fig2, ax2 = plt.subplots()
    ax2.imshow(wordcloud)
    ax2.axis("off")
    st.pyplot(fig2)

# Function to evaluate model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    pred_prob_train = model.predict_proba(X_train)[:, 1]
    pred_prob_test = model.predict_proba(X_test)[:, 1]

    roc_auc_train = roc_auc_score(y_train, y_pred_train)
    roc_auc_test = roc_auc_score(y_test, y_pred_test)

    st.subheader("ROC Curve")
    fpr_train, tpr_train, _ = roc_curve(y_train, pred_prob_train)
    fpr_test, tpr_test, _ = roc_curve(y_test, pred_prob_test)
    fig3, ax3 = plt.subplots()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_train, tpr_train, label="Train ROC AUC: {:.2f}".format(roc_auc_train))
    plt.plot(fpr_test, tpr_test, label="Test ROC AUC: {:.2f}".format(roc_auc_test))
    plt.legend()
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    st.pyplot(fig3)

    cm_train = confusion_matrix(y_train, y_pred_train)
    cm_test = confusion_matrix(y_test, y_pred_test)
    fig4, ax4 = plt.subplots(1, 2, figsize=(11, 4))
    sns.heatmap(cm_train, annot=True, xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], cmap="Oranges", fmt='.4g', ax=ax4[0])
    ax4[0].set_title("Train Confusion Matrix")
    sns.heatmap(cm_test, annot=True, xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], cmap="Oranges", fmt='.4g', ax=ax4[1])
    ax4[1].set_title("Test Confusion Matrix")
    st.pyplot(fig4)

    st.subheader("Classification Report (Train)")
    st.text(classification_report(y_train, y_pred_train))
    st.subheader("Classification Report (Test)")
    st.text(classification_report(y_test, y_pred_test))

# Function for Spam Detection
def detect_spam(email_text, clf):
    prediction = clf.predict([email_text])
    if prediction == 0:
        return "This is a Ham Email!"
    else:
        return "This is a Spam Email!"

# Main Function to Run Streamlit App
def main():
    st.title("Email Spam Detection System")
    df = load_data()

    # Sidebar Navigation
    option = st.sidebar.selectbox("Choose Option", ["Overview", "Visualizations", "Train Model", "Spam Detection"])

    if option == "Overview":
        visualize_data(df)

    if option == "Visualizations":
        visualize_data(df)

    if option == "Train Model":
        st.subheader("Model Training and Evaluation")
        X_train, X_test, y_train, y_test = train_test_split(df.Message, df.Spam, test_size=0.25)
        clf = Pipeline([
            ('vectorizer', CountVectorizer()), 
            ('nb', MultinomialNB())  
        ])
        evaluate_model(clf, X_train, X_test, y_train, y_test)
        st.success("Model Training and Evaluation Completed")

    if option == "Spam Detection":
        st.subheader("Email Spam Detection")
        input_email = st.text_area("Enter the Email Content Here...")
        if st.button("Detect"):
            X_train, X_test, y_train, y_test = train_test_split(df.Message, df.Spam, test_size=0.25)
            clf = Pipeline([
                ('vectorizer', CountVectorizer()), 
                ('nb', MultinomialNB())  
            ])
            clf.fit(X_train, y_train)
            result = detect_spam(input_email, clf)
            st.success(result)

if __name__ == "__main__":
    main()