import streamlit as st
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

st.title("📊 Analisis dan Klasifikasi Sentimen Komentar signal")

uploaded_file = st.file_uploader("📁 Upload file CSV hasil scraping", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Pratinjau Data Awal")
    st.dataframe(df.head())

    # Drop duplikat dan data null
    df.drop_duplicates(subset='userName', keep='first', inplace=True)
    df.dropna(inplace=True)

    if 'at' in df.columns:
        df['at'] = pd.to_datetime(df['at'])
        df['year'] = df['at'].dt.year

        # Visualisasi sentimen per tahun
        st.subheader("📆 Distribusi Sentimen Berdasarkan Tahun")
        for tahun in [2021, 2025]:
            df_tahun = df[df['year'] == tahun]
            if not df_tahun.empty:
                sentiment_counts = df_tahun['sentiment'].value_counts()
                st.write(f"*Sentimen di Tahun {tahun}:*")
                st.bar_chart(sentiment_counts)
            else:
                st.write(f"Tidak ada data untuk tahun {tahun}")

    # Cek kolom untuk klasifikasi
    if 'content' in df.columns and 'sentiment' in df.columns:
        st.subheader("🤖 Klasifikasi Sentimen Komentar")

        df_class = df[['content', 'sentiment']]
        X = df_class['content']
        y = df_class['sentiment']

        # Fungsi pembersih teks
        def clean_text(text):
            text = text.lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            tokens = text.split()
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(word) for word in tokens]
            return ' '.join(tokens)

        X_clean = X.apply(clean_text)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42)

        # TF-IDF
        vectorizer = TfidfVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Train model
        model = MultinomialNB()
        model.fit(X_train_vec, y_train)

        # Evaluasi
        y_pred = model.predict(X_test_vec)
        st.write("📄 *Classification Report*")
        st.text(classification_report(y_test, y_pred))

        accuracy = accuracy_score(y_test, y_pred)
        st.metric("🎯 Akurasi Model", f"{accuracy:.2%}")

        # Prediksi konten baru
        st.subheader("📥 Prediksi Sentimen Komentar Baru")
        new_input = st.text_area("Tulis komentar baru di bawah ini")
        if new_input:
            cleaned_input = clean_text(new_input)
            vec_input = vectorizer.transform([cleaned_input])
            prediction = model.predict(vec_input)
            st.success(f"Prediksi Sentimen: *{prediction[0]}*")
    else:
        st.error("Kolom 'content' dan 'sentiment' tidak ditemukan di dataset.")