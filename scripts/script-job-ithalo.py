import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import DoubleType

# Importações ML
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, IndexToString, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.clustering import KMeans

# --- 1. CONFIGURAÇÃO ---
spark = SparkSession.builder.appName("ALSAnalysisColab").getOrCreate()

print("--- INICIANDO PIPELINE (CORRIGIDO PARA DADOS CATEGÓRICOS) ---")

# --- 2. CAMINHOS ---
path_entrada = "/content/dados_tratados (1).parquet"
path_saida = "/content/dataset_final_5classes_v2/"

# --- 3. LEITURA ---
try:
    df = spark.read.parquet(path_entrada)
except Exception as e:
    print(f"Erro crítico na leitura: {e}")
    sys.exit(1)

# --- 4. PREPARAÇÃO ESPECÍFICA (AQUI ESTAVA O ERRO) ---
print("--> Tratando categorias textuais...")

# CORREÇÃO CRÍTICA: Mapeamento Manual de 'progressionrate' (String -> Número Ordinal)
# Isso é essencial para o K-Means entender a ordem de gravidade
df_limpo = df.withColumn("progressionrate_num",
    when(col("progressionrate") == "slow", 1.0)
    .when(col("progressionrate") == "intermediate", 2.0)
    .when(col("progressionrate") == "fast", 3.0)
    .otherwise(0.0) # Caso apareça algo estranho
)

# Remoção de Nulos (Baseado nas colunas originais)
cols_essenciais = ["gene", "survivalmonths", "motorsymptomsscore", "ageofonset", "sex", "progressionrate"]
df_limpo = df_limpo.na.drop(subset=cols_essenciais)

print(f"--> Linhas válidas após limpeza: {df_limpo.count()}")

if df_limpo.count() == 0:
    print("ERRO FATAL: Dataset vazio. Verifique os nomes das colunas.")
    sys.exit(1)

# ==========================================
# PARTE A: IA (ESTADIAMENTO CLÍNICO)
# ==========================================
print("--> Executando Parte A: IA (Random Forest)...")

# Target
df_limpo = df_limpo.withColumn("target_estagio_real",
    when(col("survivalmonths") < 24, "1_Agudo")
    .when(col("survivalmonths") >= 48, "3_Cronico")
    .otherwise("2_Intermediario")
)

# Engenharia de Features
# Adicionamos 'progressionrate' nas categóricas pois é texto ("fast", "slow"...)
cols_cat = ["sex", "variant", "gene", "onsetsite", "progressionrate"]
stages = []

for coluna in cols_cat:
    # HandleInvalid='keep' garante que o valor 'None' no gene não quebre o código
    indexer = StringIndexer(inputCol=coluna, outputCol=f"{coluna}_idx", handleInvalid="keep")
    encoder = OneHotEncoder(inputCols=[indexer.getOutputCol()], outputCols=[f"{coluna}_vec"], dropLast=False)
    stages += [indexer, encoder]

indexer_target = StringIndexer(inputCol="target_estagio_real", outputCol="label_estagio").fit(df_limpo)
stages += [indexer_target]

# Features Numéricas
cols_num = ["ageofonset", "motorsymptomsscore"]
input_final = cols_num + [f"{c}_vec" for c in cols_cat]

assembler = VectorAssembler(inputCols=input_final, outputCol="features")
stages += [assembler]

# Treino
pipeline = Pipeline(stages=stages)
model_prep = pipeline.fit(df_limpo)
df_prep = model_prep.transform(df_limpo)

rf = RandomForestClassifier(featuresCol="features", labelCol="label_estagio", predictionCol="pred_idx", numTrees=100, seed=42)
model_rf = rf.fit(df_prep)
df_resultado = model_rf.transform(df_prep)

# Tradução
if "previsao_ia_estagio" in df_resultado.columns:
    df_resultado = df_resultado.drop("previsao_ia_estagio")
converter = IndexToString(inputCol="pred_idx", outputCol="previsao_ia_estagio", labels=indexer_target.labels)
df_final = converter.transform(df_resultado)

# ==========================================
# PARTE B: CÁLCULO (VELOCIDADE DE DANO)
# ==========================================
print("--> Executando Parte B: Cálculo Matemático...")

df_final = df_final.withColumn("velocidade_bruta_calc",
    col("motorsymptomsscore") / (col("survivalmonths") + 0.1)
)

try:
    cortes = df_final.stat.approxQuantile("velocidade_bruta_calc", [0.2, 0.4, 0.6, 0.8], 0.01)
    c20, c40, c60, c80 = cortes
except:
    c20, c40, c60, c80 = 0.1, 0.2, 0.3, 0.4

df_final = df_final.withColumn("classificacao_velocidade_5niveis",
    when(col("velocidade_bruta_calc") < c20, "1_Muito_Lenta")
    .when(col("velocidade_bruta_calc") < c40, "2_Lenta")
    .when(col("velocidade_bruta_calc") < c60, "3_Media")
    .when(col("velocidade_bruta_calc") < c80, "4_Rapida")
    .otherwise("5_Fulminante")
)

# ==========================================
# PARTE C: CLUSTERIZAÇÃO (K-MEANS K=4)
# ==========================================
print("--> Executando Parte C: Clusterização K-Means...")

# 1. Seleção de Colunas para Cluster
# Usamos 'progressionrate_num' (1.0, 2.0, 3.0) que criamos lá em cima
# Usamos 'gene_idx' que foi criado pelo StringIndexer da Parte A
cols_cluster = ["progressionrate_num", "motorsymptomsscore", "gene_idx", "ageofonset"]

assembler_k = VectorAssembler(inputCols=cols_cluster, outputCol="features_cluster_raw")
df_cluster_vec = assembler_k.transform(df_final)

scaler = StandardScaler(inputCol="features_cluster_raw", outputCol="features_cluster_scaled", 
                        withStd=True, withMean=True)
scaler_model = scaler.fit(df_cluster_vec)
df_cluster_scaled = scaler_model.transform(df_cluster_vec)

kmeans = KMeans(featuresCol="features_cluster_scaled", predictionCol="cluster_clinico", k=4, seed=42)
model_kmeans = kmeans.fit(df_cluster_scaled)
df_final_completo = model_kmeans.transform(df_cluster_scaled)

# ==========================================
# SALVAMENTO
# ==========================================
colunas_finais = [
    "caseid", "gene", "variant", "sex", "ageofonset",
    "motorsymptomsscore", "survivalmonths", "progressionrate",
    "target_estagio_real", "previsao_ia_estagio",
    "classificacao_velocidade_5niveis",
    "cluster_clinico"
]

print(f"--> Salvando {df_final_completo.count()} registros.")
df_final_completo.select(colunas_finais).write.mode("overwrite").parquet(path_saida)

print("--- JOB FINALIZADO COM SUCESSO ---")
spark.stop()