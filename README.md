# Classificação de Relevância de eventos
Este repositório contém um script didático no formato do jupyter notebook para classificação de eventos relevantes.

O problema inicial é o de classificar a relevância de ocorrências diárias na Itaipu Binacional, estes eventos relevantes seguem determinados padrões, como por exemplo:

    - Problemas no elevador sem pessoas presas não é um evento relevante
    - Problemas no elevador com pessoas presas é um evento relevante

Este tipo de ocorrência se repete diversas vezes, e a ideia do algoritmo é treinar um modelo de machine learning que classifique estes eventos como relevantes ou não.

Para teste deste modelo, foram utilizados relatórios de eventos relevantes (documentos em pdf), juntamente com uma api interna da Itaipu que faz a requisição de todas as ocorrências em um intervalo de tempo, combinando os eventos relevantes dos documentos em pdf com a api de todas as ocorrências, foi obtido um banco de dados com mais de 5700 ocorrências classificadas como relevantes ou não.

Para o treino do modelo, inicialmente se faz necessário uma vetorização dos textos, e para isso foi utilizado o método TF-IDF (Term Frequency-Inverse Document Frequency), este método consiste em atribuir pesos a cada palavra das Descrições, considerando sua relevância local (ocorrência por ocorrência) e sua relevância global (em todas as ocorrências do banco de dados).

Vetorizado todas as descrições, foi implementado o método de Regressão Logística, ideal para classificação binária de eventos.

## Pré-processamento dos dados
Para uma melhor eficiência do treinamento do modelo, incialmente foi realizado um processamento dos dados brutos:
    
1. Remoção dos stopwords: stopwords são palavras muito comuns que geralmente são filtradas durante o pré-processamento de texto em tarefas de processamento de linguagem natural. Essas palavras têm pouca relevância semântica para a análise de texto e podem ser removidas com segurança sem prejudicar o significado geral do texto. Exemplos de stopwords em português incluem palavras como "o", "a", "de", "para", "com", "em", "que", "se", entre outras.

2. Remoção de caracteres especiais e acentos das palavras.

3. Passar todas as letras dos textos para minúsculas

## Implementação do método TF-IDF (Term Frequency-Inverse Document Frequency)

1. **Cálculo da Frequência dos Termos (TF)**:
   - Para cada palavra em um documento, calcula-se a frequência do termo, ou seja, quantas vezes a palavra aparece no documento.
   - A equação para calcular a frequência do termo é:

$$
TF_{t,d} = \frac{\text{Número de vezes que o termo } t \text{ aparece em } d}{\text{Número total de termos em } d}
$$

   - Isso resulta em um valor entre 0 e 1, indicando a frequência relativa da palavra no documento.

2. **Cálculo da Frequência do Documento Inverso (IDF)**:
   - Calcula-se o inverso da frequência de documentos que contêm o termo.
   - A equação para calcular o IDF é:
     
$$
IDF_t = \log\left(\frac{\text{Número total de documentos}}{\text{Número de documentos que contêm o termo } t}\right)
$$
  
   - O logaritmo é usado para penalizar termos que aparecem em muitos documentos.

4. **Cálculo do Produto TF-IDF**:
   - O produto TF-IDF de uma palavra é o produto da frequência do termo no documento (TF) e o inverso da frequência do documento (IDF).
   - A equação é:
     
$$
TF-IDF_{t,d} = TF_{t,d} \times IDF_t
$$
     
   - Isso resulta em um valor que representa a importância da palavra no contexto do documento e do corpus.

5. **Implementação usando TfidfVectorizer**:
   - A classe TfidfVectorizer do scikit-learn implementa a vetorização de texto usando a técnica TF-IDF.
   - Passos:
     - Criação do vetorizador: Instancie um objeto TfidfVectorizer.
     - Ajuste do vetorizador: Use o método fit_transform para ajustar o vetorizador aos dados de treinamento. Isso aprenderá o vocabulário do corpus e calculará os pesos TF-IDF para cada palavra.
     - Vetorização dos dados: Use o método transform para transformar os dados de treinamento e teste em vetores TF-IDF.

6. **Utilização dos Vetores TF-IDF**:
   - Os vetores TF-IDF resultantes são usados como entrada para modelos de aprendizado de máquina, que podem ser treinados e usados para fazer previsões.

# Implementação do método de Regressão Logistica

1. **Instanciação do Modelo de Regressão Logística**:
    - Inicialmente é criado uma instância do modelo de Regressão Logística usando a classe *LogisticRegression()* do scikit-learn. Este modelo será usado para fazer previsões sobre a relevância das descrições.

2. **Treinamento do Modelo**:
    - Após isso é ralizado o treinamento do modelo de regressão logística usando o método *fit()*. Este método recebe dois argumentos:
    - *X_train_tfidf*: Este é o conjunto de dados de treinamento pré-processado e vetorizado usando o TF-IDF. Ele contém os vetores TF-IDF das descrições de treinamento.
    - *y_train*: Este é o vetor de rótulos correspondentes aos dados de treinamento. Cada rótulo indica se a descrição correspondente é relevante (1) ou não relevante (0).
    - Durante o treinamento, o modelo ajusta os parâmetros (coeficientes) de forma a minimizar a função de perda, que é geralmente a função de entropia cruzada. Isso é feito usando técnicas de otimização, como o gradiente descendente, para ajustar os parâmetros na direção que reduz a função de perda.


3. **Previsões no conjunto de teste e avaliação do modelo**:
    - Aqui, estamos utilizando o modelo treinado para fazer previsões sobre o conjunto de teste. O método *predict()* é aplicado ao modelo, passando como entrada os vetores TF-IDF do conjunto de teste (*X_test_tfidf*), lembrando que este X é o resultado da vetorização das Descrições teste do Banco de dados. Isso retorna as previsões do modelo para cada descrição de teste.

    - Após isso, é avaliado o desempenho do modelo utilizando métricas de avaliação. é utilizado a função *accuracy_score()* do scikit-learn para calcular a acurácia do modelo. Ela compara as previsões feitas com os rótulos verdadeiros do conjunto de teste *y_test*. Essa métrica nos dá uma medida de quão preciso é o modelo em suas previsões.

