import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import warnings
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import uuid

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Аналитика отзывов", layout="wide")

if 'user_id' not in st.session_state:
    st.session_state.user_id = np.random.randint(100, 1000)

DB_NAME = 'reviews.db'

def get_db_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            product TEXT,
            rating INTEGER,
            text TEXT,
            sentiment INTEGER,
            date DATE
        )
    ''')
    conn.commit()
    
    # Генерация данных, если таблица пуста
    c.execute('SELECT COUNT(*) FROM reviews')
    if c.fetchone()[0] == 0:
        generate_custom_dataset(conn)
    conn.close()

def generate_custom_dataset(conn):
    positive_reviews = ["отлично", "замечательно", "супер", "рекомендую", "лучший", 
                        "качество отличное", "доволен", "прекрасно", "великолепно", "работает идеально"]
    negative_reviews = ["ужасно", "плохо", "разочарован", "не рекомендую", "брак", 
                        "качество плохое", "ужасное качество", "не работает", "деньги на ветер", "кошмар"]
    products = ["Смартфон", "Ноутбук", "Наушники", "Часы", "Планшет", "Монитор", "Клавиатура"]
    
    data = []
    for _ in range(250):
        rating = np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.1, 0.15, 0.3, 0.3])
        if rating >= 4:
            text = np.random.choice(positive_reviews) + " " + np.random.choice(["все супер", "рад покупке", ""])
            sentiment = 1
        elif rating <= 2:
            text = np.random.choice(negative_reviews) + " " + np.random.choice(["никогда больше", "верните деньги", ""])
            sentiment = 0
        else:
            text = "нормальный товар, пойдет"
            sentiment = None
            
        data.append((
            np.random.randint(1, 31),
            np.random.choice(products), 
            int(rating),
            text,
            sentiment,
            (datetime.now() - timedelta(days=np.random.randint(0, 60))).date()
        ))
        
    c = conn.cursor()
    c.executemany('INSERT INTO reviews (user_id, product, rating, text, sentiment, date) VALUES (?, ?, ?, ?, ?, ?)', data)
    conn.commit()

@st.cache_resource
def train_models(df_binary):
    X = df_binary['text'].values
    y = df_binary['sentiment'].values
    
    vectorizer = TfidfVectorizer(max_features=100)
    X_vec = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    }
    
    metrics = []
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        trained_models[name] = model
        
        metrics.append({
            'Модель': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0)
        })
        
    metrics_df = pd.DataFrame(metrics).set_index('Модель')
    best_model_name = metrics_df['F1'].idxmax()
    return vectorizer, trained_models, metrics_df, best_model_name

def predict_sentiment(text, vectorizer, model):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0][int(pred)]
    return "Позитивный" if pred == 1 else "Негативный", proba

def get_top_products_positive(df):
    df_pos = df[df['sentiment'].notna()]
    stats = df_pos.groupby('product')['sentiment'].mean().sort_values(ascending=False) * 100
    return stats

def get_top_words_negative(df):
    neg_texts = df[df['sentiment'] == 0]['text'].dropna()
    if len(neg_texts) == 0: return []
    vec = CountVectorizer(max_features=5, stop_words=['и', 'в', 'на', 'не', 'что'])
    vec.fit(neg_texts)
    return vec.get_feature_names_out()

def get_recommendations_cf(df, current_user_id, top_n=3):
    user_item_matrix = df.pivot_table(index='user_id', columns='product', values='rating').fillna(0)
    
    if current_user_id not in user_item_matrix.index:
        return df.groupby('product')['rating'].mean().nlargest(top_n).index.tolist()
    
    user_sim = cosine_similarity(user_item_matrix)
    user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)
    
    similar_users = user_sim_df[current_user_id].sort_values(ascending=False).index[1:]
    
    user_rated_products = user_item_matrix.loc[current_user_id]
    unrated_products = user_rated_products[user_rated_products == 0].index
    
    recommendations = {}
    for prod in unrated_products:
        score = 0
        weight_sum = 0
        for sim_user in similar_users:
            if user_item_matrix.loc[sim_user, prod] > 0:
                score += user_sim_df.loc[current_user_id, sim_user] * user_item_matrix.loc[sim_user, prod]
                weight_sum += user_sim_df.loc[current_user_id, sim_user]
        if weight_sum > 0:
            recommendations[prod] = score / weight_sum
            
    sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    return [prod for prod, score in sorted_recs][:top_n]

def create_and_save_report(df, metrics_df):
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Распределение оценок
    ax1 = plt.subplot(2, 2, 1)
    df['rating'].value_counts().sort_index().plot(kind='bar', color='royalblue', ax=ax1)
    ax1.set_title('Распределение оценок')
    
    # 2. Динамика по дням
    ax2 = plt.subplot(2, 2, 2)
    daily = df.groupby('date').size()
    daily.plot(kind='line', marker='o', color='green', ax=ax2)
    ax2.set_title('Динамика отзывов')
    
    # 3. Доля позитива по товарам
    ax3 = plt.subplot(2, 2, 3)
    top_pos = get_top_products_positive(df)
    top_pos.plot(kind='bar', color='orange', ax=ax3)
    ax3.set_title('Доля позитивных отзывов (%)')
    
    # 4. Метрики моделей
    ax4 = plt.subplot(2, 2, 4)
    metrics_df[['Accuracy', 'F1']].plot(kind='bar', ax=ax4)
    ax4.set_title('Сравнение моделей')
    
    plt.tight_layout()
    plt.savefig('student_report.png', dpi=150, bbox_inches='tight')
    return fig

init_db()
conn = get_db_connection()
df = pd.read_sql('SELECT * FROM reviews', conn)
df['date'] = pd.to_datetime(df['date']).dt.date
conn.close()

df_binary = df[df['sentiment'].notna()].copy()
vectorizer, trained_models, metrics_df, best_model_name = train_models(df_binary)
best_model = trained_models[best_model_name]

st.sidebar.title("Навигация")
page = st.sidebar.radio("Перейти к:", ["Главная (Оставить отзыв)", "Статистика и Аналитика"])
st.sidebar.write("---")
st.sidebar.write(f"Текущий User ID: **{st.session_state.user_id}**")

if page == "Главная (Оставить отзыв)":
    st.title("Система отзывов и рекомендаций")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Оставить новый отзыв")
        with st.form("review_form"):
            products_list = df['product'].unique().tolist()
            selected_product = st.selectbox("Выберите товар", products_list)
            rating = st.slider("Ваша оценка", 1, 5, 5)
            text = st.text_area("Текст отзыва")
            submitted = st.form_submit_button("Отправить отзыв")
            
            if submitted and text:
                sentiment_val = None
                if rating >= 4: sentiment_val = 1
                elif rating <= 2: sentiment_val = 0
                
                conn = get_db_connection()
                c = conn.cursor()
                c.execute('INSERT INTO reviews (user_id, product, rating, text, sentiment, date) VALUES (?, ?, ?, ?, ?, ?)',
                          (st.session_state.user_id, selected_product, rating, text, sentiment_val, datetime.now().date()))
                conn.commit()
                conn.close()
                
                pred_label, proba = predict_sentiment(text, vectorizer, best_model)
                st.success(f"Отзыв сохранён! Нейросеть оценила текст как: **{pred_label}** (уверенность {proba:.1%})")
                st.rerun()

    with col2:
        st.subheader("Специально для вас")
        recs = get_recommendations_cf(df, st.session_state.user_id)
        if recs:
            for rec in recs:
                st.info(f"**{rec}**")
        else:
            st.write("Пока нет рекомендаций. Оставьте пару отзывов!")

elif page == "Статистика и Аналитика":
    st.title("Аналитика магазина")
    
    st.header("1. Сравнение ML моделей")
    st.dataframe(metrics_df.style.format("{:.2%}"))
    
    st.markdown(f"**Вывод по моделям:** На основе метрики F1-score лучшей моделью оказалась **{best_model_name}**. "
                f"Мы используем её для классификации новых отзывов, так как она лучше всего балансирует точность и полноту поиска негатива/позитива.")
    
    st.header("2. Инсайты из отзывов")
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("**Топ товаров по доле позитива:**")
        top_pos = get_top_products_positive(df)
        st.dataframe(top_pos.map("{:.1f}%".format))
        
    with col4:
        st.write("**Топ-5 частых слов в негативе:**")
        top_words = get_top_words_negative(df)
        for word in top_words:
            st.error(word)
            
    st.header("3. Облако слов")
    all_text = " ".join(df['text'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc)
    
    st.header("4. Отчет и Визуализация")
    fig_report = create_and_save_report(df, metrics_df)
    st.pyplot(fig_report)
    st.success("Все графики успешно сохранены в файл `student_report.png`")

st.header("Все отзывы")
with st.expander("Нажмите, чтобы развернуть таблицу со всеми отзывами"):
    df_display = df.sort_values(by='date', ascending=False)
    
    st.dataframe(
        df_display[['date', 'product', 'rating', 'text', 'user_id', 'sentiment']],
        use_container_width=True,
        column_config={
            "sentiment": st.column_config.CheckboxColumn("Позитив?"),
            "date": st.column_config.DateColumn("Дата"),
            "rating": st.column_config.NumberColumn("Оценка", format="%d ⭐")
        }
    )

    csv = df_display.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="Скачать все отзывы в CSV",
        data=csv,
        file_name='all_reviews.csv',
        mime='text/csv',
    )