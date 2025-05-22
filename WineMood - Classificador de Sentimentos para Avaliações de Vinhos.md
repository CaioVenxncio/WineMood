# WineMood - Classificador de Sentimentos para Avaliações de Vinhos

WineMood é um sistema inteligente que analisa avaliações textuais de vinhos e classifica automaticamente o sentimento expresso como positivo, neutro ou negativo. Utilizando técnicas de Processamento de Linguagem Natural (NLP) combinadas com Deep Learning, o projeto transforma a linguagem humana em insights úteis para consumidores e empresas.

## 📋 Visão Geral

Este classificador pode ser integrado em plataformas de e-commerce de vinhos, como a Vinheria Agnello, oferecendo aos clientes uma análise automática de sentimentos das avaliações antes de fazer uma compra.

### 🎯 Objetivos

- Aplicar técnicas de NLP e Deep Learning a dados do mundo real
- Construir um modelo capaz de entender e classificar emoções em texto
- Desenvolver uma interface web amigável para testes em tempo real
- Apresentar uma prova de conceito prática e aplicável para empresas

## 🧰 Tecnologias e Ferramentas

| Categoria | Tecnologia |
|-----------|------------|
| Linguagem de Programação | Python |
| Framework de Deep Learning | TensorFlow + Keras |
| Manipulação de Dados | Pandas, NumPy |
| NLP & Pré-processamento | NLTK, Regex, Tokenizer |
| Interface Web | Streamlit |
| Dataset | Wine Reviews (Kaggle) |
| Plataforma de Treinamento | Google Colab (com GPU gratuita) |
| Visualização | Matplotlib / Streamlit |
| Controle de Versão | GitHub |

## 📦 Dataset

- **Nome**: Wine Reviews Dataset
- **Fonte**: Kaggle (https://www.kaggle.com/datasets/zynicide/wine-reviews)
- **Registros**: ~130.000 avaliações de vinhos
- **Colunas utilizadas**:
  - `description`: texto da avaliação do vinho
  - `points`: pontuação do vinho (de 80 a 100)
- **Rótulo derivado**: Sentimento (positivo, neutro, negativo) baseado na pontuação

## 🔧 Etapas de Desenvolvimento

### 1. Coleta e Exploração de Dados
- Carregamento do dataset usando pandas
- Verificação de valores nulos, distribuição de classes e estatísticas descritivas

### 2. Criação de Rótulos
- Conversão de pontuações numéricas em categorias de sentimento:
  - ≥ 90: Positivo
  - 80–89: Neutro
  - < 80: Negativo

### 3. Pré-processamento de Texto
- Limpeza de texto (minúsculas, remoção de pontuação e stopwords)
- Tokenização e padronização de sequências para comprimento fixo

### 4. Treinamento do Modelo de Deep Learning
- **Arquitetura do modelo**:
  - Embedding para vetorização de texto
  - LSTM para compreensão de sequência
  - Dense com softmax para classificação
- Divisão treino/teste (80/20)
- Métricas: acurácia, perda

### 5. Avaliação
- Avaliação do modelo no conjunto de teste usando `.evaluate()`
- Impressão da acurácia e análise de predições incorretas

### 6. Interface Web (Streamlit)
- Interface web onde os usuários podem inserir avaliações de texto
- O modelo prevê e exibe o sentimento detectado
- Funcionalidades extras:
  - Gráfico de barras com percentuais de sentimentos
  - Processamento de múltiplas avaliações simultaneamente
  - Aba "Sobre o projeto" com explicações detalhadas

## 🖼 Exemplo de Uso

**Entrada do Usuário**:
"Great complexity with floral aromas and a long finish."

**Saída Prevista**:
Sentimento: Positivo

## 🌍 Aplicações no Mundo Real

- Recomendação automática de vinhos com base no sentimento do cliente
- Auxílio a vendedores e produtores de vinho para entender a percepção do público
- Destaque de vinhos bem avaliados em plataformas de e-commerce

## 📈 Resultados Esperados

- Acurácia acima de 80% com dados de treinamento balanceados
- Interface web leve e responsiva
- Código modular e reutilizável para tarefas similares de análise de sentimentos

## 🚀 Destaques do Projeto

- Projeto totalmente funcional com interface em tempo real
- Utiliza dados do mundo real com aplicações práticas
- Combina NLP, LSTM e implantação Web
- Pronto para integração em plataformas comerciais

## 📝 Instruções de Uso

### Execução Local

1. Clone o repositório
2. Instale as dependências: `pip install -r requirements.txt`
3. Execute o aplicativo Streamlit: `streamlit run app.py`

### Google Colab

1. Abra o notebook `colab_notebook.ipynb` no Google Colab
2. Execute todas as células para treinar o modelo e gerar os arquivos necessários
3. Baixe os arquivos gerados (model.h5, tokenizer.pickle) para uso local

### Deploy Online

O aplicativo pode ser implantado no Streamlit Cloud para acesso público:

1. Crie uma conta no Streamlit Cloud
2. Conecte seu repositório GitHub contendo os arquivos do projeto
3. Configure o deploy com o arquivo principal como app.py

## 👥 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests com melhorias.

## 📄 Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.
