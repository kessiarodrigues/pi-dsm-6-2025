# Relatório de pré processamento do pi 6 semestre
### Este documento tem como finalidade explicar como foi realizado o pré processamento da base do pi

Aluno: Késsia Rodrigues Borges

A base utilizada é uma base com dados de pessoas diabéticas(2), não diabéticas(0) e pré diabéticas(1)
O objetivo do trabalho final é obter uma recomendação confiável, onde com base nos dados inputados pelo usuário, possa classificá-lo e recomendar bons hábitos

---

### 1. Inicialização e definição de atributos

```python
class PreProcessor:
    def __init__(self, path='diabetes_012_health_indicators_BRFSS2015.csv'):
        self.path = path
        self.colunas_irrelevantes = ['AnyHealthcare', 'NoDocbcCost', 'Income', 'Education']
        self.scaler = StandardScaler()
        self.df = pd.read_csv(self.path)
```

* **`self.path`**: caminho do arquivo de entrada.
* **`self.colunas_irrelevantes`**: lista das features que serão removidas por baixa relevância, baseada em análise de correlação.
* **`self.scaler`**: instância do `StandardScaler`, que normaliza as features para média 0 e desvio padrão 1.
* **`self.df`**: DataFrame com os dados carregados do arquivo CSV.

---

### 2. Análise de correlação

```python
def correlation(self):
    correlacao = self.df.corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlacao, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title('Matriz de Correlação da Base de Diabetes')
    plt.show()
```

* **`self.df.corr()`**: calcula a matriz de correlação de Pearson entre todas as colunas numéricas do DataFrame.
* **Heatmap**: visualiza rapidamente relações lineares entre atributos e a variável alvo (`Diabetes_012`), embasando a exclusão de variáveis irrelevantes.

---

### 3. Pré-processamento e limpeza

```python
def pre_process(self, arquivo_saida):
    self.correlation()
    self.df = self.df.drop(columns=self.colunas_irrelevantes)
    valores_nulos = self.df.isnull().sum()
    print(valores_nulos)
    X = self.df.drop('Diabetes_012', axis=1)
    y = self.df['Diabetes_012']
    X_norm = self.scaler.fit_transform(X)
    self.df_normalizado = pd.DataFrame(X_norm, columns=X.columns)
    self.df_normalizado['Diabetes_012'] = y.reset_index(drop=True)
    self.df_normalizado.to_csv(arquivo_saida, index=False)
```

#### **Passos detalhados:**

1. **`self.correlation()`**: mostra a matriz de correlação antes da remoção das colunas.
2. **Remoção das colunas irrelevantes**: `self.df.drop(columns=self.colunas_irrelevantes)` elimina variáveis que não contribuem para a predição.
3. **Checagem de valores nulos**: `self.df.isnull().sum()` exibe a contagem de nulos por coluna, auxiliando na identificação de inconsistências.
4. **Separação de variáveis**:

   * `X = self.df.drop('Diabetes_012', axis=1)`: atributos preditores.
   * `y = self.df['Diabetes_012']`: variável alvo.
5. **Normalização**:
   * Esse processo foi executado apenas depois dos dados tratados já exportados para um csv, já que para algumas partes do processo, é necessário o valor categórico da coluna (0 e 1)  
   * `self.scaler.fit_transform(X)` ajusta os dados para média 0 e desvio padrão 1, fundamental para métodos baseados em distância como KNN.
6. **Criação do DataFrame final**:

   * Junta os dados normalizados e a coluna alvo em um novo DataFrame.
7. **Exportação**:

   * `to_csv(arquivo_saida, index=False)` salva o resultado em CSV pronto para modelagem.



# Explicação do Processo de Extração de Padrões e Importância das Variáveis

Nesta etapa do projeto, foram aplicadas técnicas de **clusterização (KMeans)**, **extração de regras de associação (Apriori)** e análise de **importância de variáveis** para identificar padrões relevantes e fatores de risco associados ao diabetes na base de dados.

---

## 1. Clusterização com KMeans

O algoritmo KMeans foi utilizado para agrupar os indivíduos em três clusters, correspondendo às classes: não diabético, pré-diabético e diabético.

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_norm)
df_clusters = pd.DataFrame(X_norm, columns=X.columns)
df_clusters['Cluster'] = clusters

print(df_clusters['Cluster'].value_counts())
```

---

## 2. Extração de Regras de Associação (Apriori)

Para aplicar o Apriori, as variáveis foram binarizadas considerando a mediana como limiar. Assim, foi possível identificar padrões frequentes de características nos dados.

```python
from mlxtend.frequent_patterns import apriori, association_rules

X_bin = X.copy()
for col in X_bin.columns:
	X_bin[col] = (X_bin[col] > X_bin[col].median()).astype(int)

frequent_itemsets = apriori(X_bin, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
print(rules.head())
```

---

## 3. Visualização das Regras e dos Clusters

As regras de associação foram visualizadas em formato de árvore para facilitar a interpretação, e os clusters foram projetados em 2D usando PCA.

```python
def plot_association_rules_tree(rules, max_rules=10):
	# ... (função conforme implementada)
plot_association_rules_tree(rules)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_norm)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.5, s=5)
plt.title('Visualização dos Clusters (PCA 2D)')
plt.show()
```

---

## 4. Importância das Variáveis

Para identificar os atributos mais relevantes na classificação, foi utilizada a propriedade `feature_importances_` do RandomForest. O método SHAP, apesar de mais interpretável, foi descartado devido ao tempo excessivo de processamento em conjuntos de dados grandes.

```python
model = RandomForestClassifier(max_depth=None, random_state=32).fit(X_train_bal, y_train_bal)
importances = model.feature_importances_
importancia_df = pd.DataFrame({
	'feature': X.columns,
	'importance': importances
}).sort_values(by='importance', ascending=False)

plt.barh(importancia_df['feature'], importancia_df['importance'])
plt.xlabel('Importância')
plt.title('Importância das Features - RandomForest')
plt.show()
```

> **Observação:**  
> O método SHAP foi testado, porém não concluiu a execução devido ao alto custo computacional. Por isso, optou-se pelo uso do `feature_importances_` do RandomForest, que forneceu resultados rápidos e satisfatórios para a análise de importância dos atributos.

---

## **Resumo**

O processo de extração de padrões permitiu identificar grupos de risco, padrões de comportamento e os fatores mais relevantes para a classificação do diabetes, subsidiando recomendações personalizadas e ações preventivas.