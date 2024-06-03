import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from kmodes.kmodes import KModes
import statsmodels.api as sm
from statsmodels.formula.api import ols


df = pd.read_csv('Ecommerce_DBS.csv', encoding='utf-8')

# Removendo colunas que não serão relevantes na análise dos dados
df = df.drop(columns=['Country', 'State', 'Latitude', 'Longituide'])

# Removendo missing values
df = df.dropna()

#####################################################################
# Resposta para a questão:                                          #
# Quais os produtos mais vendidos considerando os últimos 3 anos?   #
#####################################################################

# Convertendo a coluna Purchase Date para o formato datetime
df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], format='%d/%m/%Y')

# Selecionando os dados referentes aos últimos 3 anos
dados_recentes = df[df['Purchase Date'] >= df['Purchase Date'].max() - pd.DateOffset(years=3)]

# Somando as quantidades de cada produto no período de 3 anos
vendas = dados_recentes.groupby('Product Category')['Quantity'].sum()

# Identificando os 3 produtos mais vendidos
produtos_mais_vendidos = vendas.nlargest(3)

# Printando a resposta: Clothing (182.696), Books (181.069), Electronics (121.867)
print('Produtos mais vendidos nos últimos 3 anos (Categoria e quantidade):\n\n' + str(produtos_mais_vendidos))

#################################################
# Resposta para a questão:                      #
# Qual o produto mais caro e o mais barato?     #
#################################################

# Obtendo o índice do produto com maior preço na base de dados
maior_preco_idx = df['Product Price'].idxmax()

# Obtendo o produto mais caro da base de dados
produto_mais_caro = df.loc[maior_preco_idx]

# Printando os dados do produto mais caro
print('\nProduto mais caro:\n\n' + str(produto_mais_caro))

# Obtendo o índice do produto com menor preço na base de dados
menor_preco_idx = df['Product Price'].idxmin()

# Obtendo o menor preço na base de dados
produto_mais_barato = df.loc[menor_preco_idx]

# Printando os dados do produto mais barato
print('\nProduto mais barato:\n\n' + str(produto_mais_barato))

#####################################################################################################
# Resposta para as questões:                                                                        #
# Qual a categoria de produto mais vendida e menos vendida? Qual a categoria mais e menos cara?     #
#####################################################################################################

# Obtendo as quantidades vendidas por categoria
quantidades = df.groupby('Product Category')['Quantity'].sum()

# Obtendo a categoria mais vendida
categoria_mais_vendida = quantidades.idxmax()

# Printando a categoria mais vendida
print('\nCategoria mais vendida: ' + categoria_mais_vendida)

# Obtendo a categoria menos vendida
categoria_menos_vendida = quantidades.idxmin()

# Printando a categoria menos vendida
print('\nCategoria menos vendida: ' + categoria_menos_vendida)

# Obtendo a média de preços por categoria
precos = df.groupby('Product Category')['Product Price'].mean()

categoria_mais_cara = precos.idxmax()

# Printando a categoria com maior média de preços
print('\nCategoria mais cara: ' + categoria_mais_cara)

# Obtendo a categoria com menor média de preços
categoria_menos_cara = precos.idxmin()

print('\nCategoria menos cara: ' + categoria_menos_cara)

#############################################
# Resposta para a questão:                  #
# Qual o produto com melhor e pior NPS?     #
#############################################

# Obtendo a média dos índices NPS por categoria 
nps_scores = df.groupby('Product Category')['NPS'].mean()

# Obtendo a categoria com melhor NPS
melhor_nps = nps_scores.idxmax()

# Printando a categoria de produtos com melhor NPS
print('\nProduto com melhor NPS: ' + melhor_nps)

# Obtendo a categoria com pior pior NPS
pior_nps = nps_scores.idxmin()

# Printando a categoria de produtos com pior NPS
print('\nProduto com pior NPS: ' + pior_nps)

############################
# Análises de correlação   #
############################

# Criando tabela de contingência entre as colunas Gender e Source
tabela_de_contingencia = pd.crosstab(df['Gender'], df['Source'])

# Obtendo valores das estatísticas qui-quadrado e p-value
chi2, p, dof, expected = stats.chi2_contingency(tabela_de_contingencia)

# Exibindo a tabela de contingência e os resultados do qui-quadrado
print("\nTabela de Contingência (Gender X Source):")
print(tabela_de_contingencia)
print("\nEstatística do Qui-Quadrado:")
print(f"Qui-Quadrado: {chi2}")
print(f"P-Valor: {p}")
# Esses resultados indicam que não deve haver correlação entre Gender e Source

# Criando tabela de contingência entre as colunas Customer Age e Source
tabela_de_contingencia = pd.crosstab(df['Customer Age '], df['Source'])

# Obtendo valores das estatísticas qui-quadrado e p-value
chi2, p, dof, expected = stats.chi2_contingency(tabela_de_contingencia)

# Exibindo a tabela de contingência e os resultados do qui-quadrado
print("\nTabela de Contingência (Customer Age X Source):")
print(tabela_de_contingencia)
print("\nEstatística do Qui-Quadrado:")
print(f"Qui-Quadrado: {chi2}")
print(f"P-Valor: {p}")
# Esses resultados indicam que deve haver forte correlação entre Customer Age e Source

# Criando tabela de contingência entre as colunas Gender e Product Category
tabela_de_contingencia = pd.crosstab(df['Gender'], df['Product Category'])

# Obtendo valores das estatísticas qui-quadrado e p-value
chi2, p, dof, expected = stats.chi2_contingency(tabela_de_contingencia)

# Exibindo a tabela de contingência e os resultados do qui-quadrado
print("\nTabela de Contingência (Gender X Product Category):")
print(tabela_de_contingencia)
print("\nEstatística do Qui-Quadrado:")
print(f"Qui-Quadrado: {chi2}")
print(f"P-Valor: {p}")
# Esses resultados indicam que não deve haver correlação entre Gender e Product Category

# Criando tabela de contingência entre as colunas Customer Age e Product Category
tabela_de_contingencia = pd.crosstab(df['Customer Age '], df['Product Category'])

# Obtendo valores das estatísticas qui-quadrado e p-value
chi2, p, dof, expected = stats.chi2_contingency(tabela_de_contingencia)

# Exibindo a tabela de contingência e os resultados do qui-quadrado
print("\nTabela de Contingência (Customer Age X Product Category):")
print(tabela_de_contingencia)
print("\nEstatística do Qui-Quadrado:")
print(f"Qui-Quadrado: {chi2}")
print(f"P-Valor: {p}")
# Esses resultados indicam que não deve haver correlação entre as colunas Customer Age e Product Category

####################################################################
# Análise de Clusters para Gender, Product Category e Source       #
####################################################################

#Codificando as variáveis categóricas
df['Product Category'] = df['Product Category'].astype('category')
df['Gender'] = df['Gender'].astype('category')
df['Source'] = df['Source'].astype('category')

# Mostrar os mapeamentos
print("\nMapeamento de 'Product Category':")
print(dict(enumerate(df['Product Category'].cat.categories)))

print("\nMapeamento de 'Gender':")
print(dict(enumerate(df['Gender'].cat.categories)))

print("\nMapeamento de 'Source':")
print(dict(enumerate(df['Source'].cat.categories)))

# Convertendo as variáveis categóricas para códigos numéricos
df['Product Category'] = df['Product Category'].cat.codes
df['Gender'] = df['Gender'].cat.codes
df['Source'] = df['Source'].cat.codes

# Preparando a lista de variáveis categóricas
colunas_categoricas = ['Product Category', 'Gender', 'Source']
df_categorico = df[colunas_categoricas]

# Executando o k-modes para definir os clusters
kmodes = KModes(n_clusters=2, init='Huang', random_state=42, n_jobs=1)
clusters = kmodes.fit_predict(df_categorico)

# Adicionando os rótulos dos clusters ao DataFrame original
df['Cluster'] = clusters

# Função para adicionar jitter para melhorar a visualização dos dados
def add_jitter(arr, noise_level=0.01):
    stdev = noise_level * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev

# Adicionando jitter às colunas categóricas
df['Product Category Jitter'] = add_jitter(df['Product Category'])
df['Source Jitter'] = add_jitter(df['Source'])

# Plotando o gráfico de dispersão com jitter
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Product Category Jitter', y='Source Jitter', hue='Cluster', palette='viridis', s=100, alpha=0.6)
plt.title('Gráfico de Dispersão dos Clusters')
plt.xlabel('Product Category')
plt.ylabel('Source')
plt.legend(title='Cluster')
plt.show()

###################################################################
# ANOVA para as colunas Customer Age, Product Category e Source   #
###################################################################

# Renomeando a coluna 'Customer Age ' para 'Customer_Age'
df.rename(columns={'Customer Age ': 'Customer_Age'}, inplace=True)

# Ajustando o modelo ANOVA
model = ols('Customer_Age ~ C(df["Product Category"]) + C(df["Source"]) + C(df["Product Category"]):C(df["Source"])', data=df).fit()

# Obtendo a tabela de ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)

# Exibindo a tabela de ANOVA
print(anova_table)