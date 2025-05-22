import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
import pickle

# Configuração da página
st.set_page_config(
    page_title="WineMood - Sentiment Analyzer",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# WineMood - Analisador de Sentimentos para Avaliações de Vinhos"
    }
)

# Aplicar tema escuro personalizado
st.markdown("""
<style>
    .main {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stTextInput > div > div > input {
        background-color: #2D2D2D;
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #8A2BE2;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #9370DB;
    }
    .positive {
        color: #4CAF50;
        font-weight: bold;
        font-size: 24px;
    }
    .neutral {
        color: #9E9E9E;
        font-weight: bold;
        font-size: 24px;
    }
    .negative {
        color: #F44336;
        font-weight: bold;
        font-size: 24px;
    }
    .title {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
        color: #FFFFFF;
    }
    .subtitle {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# Função para carregar o modelo e o tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    try:
        # Carregar o modelo
        model = tf.keras.models.load_model('model.h5')
        
        # Carregar o tokenizer
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
            
        return model, tokenizer
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None, None

# Função para pré-processar o texto
def preprocess_text(text, tokenizer, max_length=100):
    # Converter para minúsculas
    text = text.lower()
    
    # Remover pontuação e caracteres especiais
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remover stopwords
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    # Tokenizar e padronizar
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    
    return padded_sequences

# Função para classificar o sentimento
def classify_sentiment(text, model, tokenizer):
    # Pré-processar o texto
    processed_text = preprocess_text(text, tokenizer)
    
    # Fazer a predição
    prediction = model.predict(processed_text)
    
    # Obter a classe com maior probabilidade
    sentiment_class = np.argmax(prediction, axis=1)[0]
    
    # Mapear para rótulos de sentimento
    sentiment_map = {0: "Negativo", 1: "Neutro", 2: "Positivo"}
    sentiment = sentiment_map[sentiment_class]
    
    # Obter as probabilidades
    probabilities = prediction[0]
    
    return sentiment, probabilities

# Função para exibir o resultado com a cor apropriada
def display_sentiment(sentiment):
    if sentiment == "Positivo":
        return st.markdown(f'<p class="positive">Sentimento: {sentiment}</p>', unsafe_allow_html=True)
    elif sentiment == "Neutro":
        return st.markdown(f'<p class="neutral">Sentimento: {sentiment}</p>', unsafe_allow_html=True)
    else:
        return st.markdown(f'<p class="negative">Sentimento: {sentiment}</p>', unsafe_allow_html=True)

# Função para plotar o gráfico de barras
def plot_sentiment_probabilities(probabilities):
    labels = ['Negativo', 'Neutro', 'Positivo']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#1E1E1E')
    ax.set_facecolor('#1E1E1E')
    
    bars = ax.bar(labels, probabilities * 100, color=['#F44336', '#9E9E9E', '#4CAF50'])
    
    # Adicionar rótulos e título
    ax.set_ylabel('Probabilidade (%)', color='white')
    ax.set_title('Distribuição de Probabilidades de Sentimento', color='white')
    
    # Personalizar eixos
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    color='white')
    
    return fig

# Interface principal
def main():
    # Carregar o modelo e o tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Título
    st.markdown('<p class="title">WineMood – Sentiment Analyzer</p>', unsafe_allow_html=True)
    
    # Criar abas
    tab1, tab2 = st.tabs(["Analisador de Sentimentos", "Sobre o Projeto"])
    
    with tab1:
        st.markdown('<p class="subtitle">Análise de Sentimentos em Avaliações de Vinhos</p>', unsafe_allow_html=True)
        
        # Área de texto para entrada do usuário
        user_input = st.text_area("Digite a descrição do vinho (ou várias descrições, uma por linha):", 
                                height=150,
                                placeholder="Ex: Great complexity with floral aromas and a long finish.")
        
        # Botão para classificar
        if st.button("Classify"):
            if user_input:
                # Dividir o texto em linhas para processar múltiplas avaliações
                reviews = user_input.strip().split('\n')
                
                # Processar cada avaliação
                results = []
                all_probabilities = []
                
                for review in reviews:
                    if review.strip():  # Verificar se a linha não está vazia
                        sentiment, probabilities = classify_sentiment(review, model, tokenizer)
                        results.append((review, sentiment))
                        all_probabilities.append(probabilities)
                
                # Exibir resultados
                if len(results) == 1:
                    # Caso de uma única avaliação
                    display_sentiment(results[0][1])
                    
                    # Plotar gráfico de probabilidades
                    st.pyplot(plot_sentiment_probabilities(all_probabilities[0]))
                else:
                    # Caso de múltiplas avaliações
                    st.markdown('<p class="subtitle">Resultados:</p>', unsafe_allow_html=True)
                    
                    # Criar DataFrame para exibir resultados
                    df_results = pd.DataFrame(results, columns=['Avaliação', 'Sentimento'])
                    
                    # Contar ocorrências de cada sentimento
                    sentiment_counts = df_results['Sentimento'].value_counts()
                    
                    # Garantir que todos os sentimentos estejam representados
                    for sentiment in ['Positivo', 'Neutro', 'Negativo']:
                        if sentiment not in sentiment_counts:
                            sentiment_counts[sentiment] = 0
                    
                    # Calcular percentuais
                    total = len(results)
                    sentiment_percentages = (sentiment_counts / total * 100).reindex(['Positivo', 'Neutro', 'Negativo'])
                    
                    # Exibir tabela de resultados
                    st.dataframe(df_results, use_container_width=True)
                    
                    # Plotar gráfico de barras com percentuais
                    fig, ax = plt.subplots(figsize=(10, 5))
                    fig.patch.set_facecolor('#1E1E1E')
                    ax.set_facecolor('#1E1E1E')
                    
                    bars = ax.bar(['Positivo', 'Neutro', 'Negativo'], 
                                sentiment_percentages, 
                                color=['#4CAF50', '#9E9E9E', '#F44336'])
                    
                    # Adicionar rótulos e título
                    ax.set_ylabel('Percentual (%)', color='white')
                    ax.set_title('Distribuição de Sentimentos', color='white')
                    
                    # Personalizar eixos
                    ax.spines['bottom'].set_color('white')
                    ax.spines['top'].set_color('white')
                    ax.spines['right'].set_color('white')
                    ax.spines['left'].set_color('white')
                    ax.tick_params(axis='x', colors='white')
                    ax.tick_params(axis='y', colors='white')
                    
                    # Adicionar valores nas barras
                    for bar in bars:
                        height = bar.get_height()
                        ax.annotate(f'{height:.1f}%',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom',
                                    color='white')
                    
                    st.pyplot(fig)
            else:
                st.warning("Por favor, digite uma avaliação de vinho para classificar.")
    
    with tab2:
        st.markdown('<p class="subtitle">Sobre o WineMood</p>', unsafe_allow_html=True)
        
        st.markdown("""
        ## O que é o WineMood?
        
        WineMood é um sistema inteligente que analisa avaliações textuais de vinhos e classifica automaticamente o sentimento expresso como positivo, neutro ou negativo. Utilizando técnicas de Processamento de Linguagem Natural (NLP) combinadas com Deep Learning, o projeto transforma a linguagem humana em insights úteis para consumidores e empresas.
        
        ## Como funciona?
        
        O sistema utiliza um modelo de Deep Learning baseado em redes neurais recorrentes (LSTM) para entender o contexto e o sentimento das avaliações de vinhos. O processo inclui:
        
        1. **Pré-processamento de texto**: limpeza, remoção de stopwords e tokenização
        2. **Vetorização**: conversão de palavras em representações numéricas
        3. **Análise de sequência**: compreensão do contexto através de LSTM
        4. **Classificação**: determinação do sentimento predominante
        
        ## Aplicações práticas
        
        - Recomendação automática de vinhos com base no sentimento do cliente
        - Auxílio a vendedores e produtores de vinho para entender a percepção do público
        - Destaque de vinhos bem avaliados em plataformas de e-commerce
        
        ## Tecnologias utilizadas
        
        - **Python**: linguagem de programação principal
        - **TensorFlow + Keras**: framework de Deep Learning
        - **NLTK**: biblioteca para processamento de linguagem natural
        - **Streamlit**: interface web interativa
        - **Pandas & NumPy**: manipulação de dados
        - **Matplotlib**: visualização de dados
        
        ## Dataset
        
        O modelo foi treinado com aproximadamente 130.000 avaliações de vinhos do Kaggle, com pontuações convertidas em categorias de sentimento:
        
        - **Positivo**: pontuação ≥ 90
        - **Neutro**: pontuação entre 80-89
        - **Negativo**: pontuação < 80
        """)

if __name__ == "__main__":
    main()
