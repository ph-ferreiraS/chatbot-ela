import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

# Bibliotecas ML do PySpark
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, ClusteringEvaluator
from pyspark.sql.functions import col, when
from pyspark.sql.types import IntegerType

# --- 1. INICIALIZAÇÃO DO GLUE ---
# Isso substitui o SparkSession.builder manual e integra com a AWS
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init('Job_ELA_Challenge', {})

print("Iniciando Job de Machine Learning no Glue...")

# --- 2. CARREGAR DADOS DO S3 ---
caminho_entrada = "s3://ela-datalake/raw/trilha_ela_dataset.csv" 
caminho_saida = "s3://ela-datalake/ml/enriched/dataset_enriquecido/"

try:
    print(f"Lendo dados de: {caminho_entrada}")
    df = spark.read.csv(caminho_entrada, header=True, inferSchema=True, sep=",")

    # Padronizar colunas (Minúsculas e sem espaços)
    for coluna_antiga in df.columns:
        nova_coluna = coluna_antiga.strip().lower()
        df = df.withColumnRenamed(coluna_antiga, nova_coluna)

    # Casting forçado de tipos numéricos importantes
    cols_to_cast = ["ageofonset", "survivalmonths", "diagnosisyear", "motorsymptomsscore"]
    for c in cols_to_cast:
        if c in df.columns:
            df = df.withColumn(c, col(c).cast(IntegerType()))

    # --- 3. PRÉ-PROCESSAMENTO ---
    print("Realizando Feature Engineering...")

    # Remover nulos nas colunas essenciais
    cols_obrigatorias = ["ageofonset", "survivalmonths", "progressionrate", "sex", "gene"]
    df_clean = df.dropna(subset=[c for c in cols_obrigatorias if c in df.columns])

    # Indexar colunas de texto (String -> Número)
    indexers = []
    cols_categoricas = ["sex", "gene", "onsetsite", "familyhistory", "progressionrate"]
    
    for c in cols_categoricas:
        if c in df_clean.columns:
            output_name = f"{c}_index" if c != "progressionrate" else "label_progression"
            indexers.append(StringIndexer(inputCol=c, outputCol=output_name))

    pipeline_preprocessing = Pipeline(stages=indexers)
    model_preprocessing = pipeline_preprocessing.fit(df_clean)
    df_transformed = model_preprocessing.transform(df_clean)

    # --- 4. ALGORITMO 1: CLUSTERIZAÇÃO (K-MEANS) ---
    # Requisito: Gerar cluster_id
    print("Treinando K-Means...")
    
    vec_assembler_cluster = VectorAssembler(inputCols=["ageofonset", "survivalmonths"], outputCol="features_cluster")
    df_cluster_ready = vec_assembler_cluster.transform(df_transformed)

    kmeans = KMeans(featuresCol="features_cluster", predictionCol="cluster_id", k=3, seed=1)
    model_kmeans = kmeans.fit(df_cluster_ready)
    predictions_cluster = model_kmeans.transform(df_cluster_ready)
    
    # Log da métrica (Aparece no CloudWatch)
    evaluator_cl = ClusteringEvaluator(featuresCol="features_cluster", predictionCol="cluster_id")
    silhouette = evaluator_cl.evaluate(predictions_cluster)
    print(f"Métrica Silhouette do Cluster: {silhouette:.4f}")

    # --- 5. ALGORITMO 2: CLASSIFICAÇÃO (RANDOM FOREST) ---
    # Requisito: Gerar predicted_score / risk_level
    print("Treinando Random Forest...")

    # Usando a versão "V2" (Melhorada) do seu colega
    cols_classificacao = [
        "ageofonset", "cluster_id", "sex_index", 
        "gene_index", "familyhistory_index", 
        "onsetsite_index", "motorsymptomsscore"
    ]
    # Garantir que só usa colunas que existem
    cols_finais = [c for c in cols_classificacao if c in predictions_cluster.columns]
    
    vec_assembler_class = VectorAssembler(inputCols=cols_finais, outputCol="features_class")
    df_class_ready = vec_assembler_class.transform(predictions_cluster)

    # Divisão Treino/Teste para avaliação interna
    train, test = df_class_ready.randomSplit([0.8, 0.2], seed=42)

    rf = RandomForestClassifier(
        labelCol="label_progression", 
        featuresCol="features_class", 
        predictionCol="predicted_class_raw", # Nome temporário
        numTrees=50, 
        maxDepth=10
    )
    
    model_rf = rf.fit(train)
    
    # Aplicar o modelo no dataset inteiro para gerar o dataset enriquecido final
    df_final_predictions = model_rf.transform(df_class_ready)

    # Avaliação simples para Log
    accuracy = MulticlassClassificationEvaluator(
        labelCol="label_progression", predictionCol="predicted_class_raw", metricName="accuracy"
    ).evaluate(df_final_predictions)
    print(f"Acurácia do modelo RF: {accuracy*100:.2f}%")

    # --- 6. FORMATAÇÃO FINAL E SALVAMENTO ---
    print("Formatando dataset final...")

    # Renomear colunas para atender especificações do desafio (predicted_score / risk_level)
    # Aqui transformamos a classe prevista numérica em algo legível ou mantemos o nome pedido
    df_enriched = df_final_predictions \
        .withColumnRenamed("predicted_class_raw", "predicted_risk_group") \
        .drop("features_cluster", "features_class", "features_raw") # Remove vetores técnicos
    
    # Selecionar colunas finais (Originais + Novas)
    # Você pode ajustar essa lista conforme necessário
    df_export = df_enriched.drop("features") 

    print(f"Salvando dataset enriquecido em: {caminho_saida}")
    
    # Gravar em Parquet (Mais rápido e eficiente para o Athena ler depois)
    df_export.write.mode("overwrite").parquet(caminho_saida)

    print("Job concluído com sucesso!")
    job.commit()

except Exception as e:
    print(f"ERRO CRÍTICO NO JOB: {str(e)}")
    sys.exit(1)
