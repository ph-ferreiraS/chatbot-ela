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
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init('Job_ELA_Challenge', {})

print("Iniciando Job de Machine Learning no Glue (Entrada Parquet)...")

# --- 2. CARREGAR DADOS DO S3 (PARQUET) ---
# Atualizado para a camada Trusted
caminho_entrada = "s3://ela-datalake/trusted/dados_tratados.parquet"
caminho_saida = "s3://ela-datalake/ml/enriched/dataset_enriquecido/"

try:
    print(f"Lendo dados TRUSTED de: {caminho_entrada}")
    
    # MUDANÇA AQUI: Lendo Parquet direto
    df = spark.read.parquet(caminho_entrada)

    # Padronizar colunas (Minúsculas e sem espaços) - Mantive por segurança
    for coluna_antiga in df.columns:
        nova_coluna = coluna_antiga.strip().lower()
        df = df.withColumnRenamed(coluna_antiga, nova_coluna)

    # (Removi a parte de converter IntegerType pois você disse que já foi feito)
    
    print(f"Esquema carregado. Total de linhas: {df.count()}")
    df.printSchema() # Ajuda a ver nos logs se os tipos vieram certos

    # --- 3. PRÉ-PROCESSAMENTO ---
    print("Realizando Feature Engineering...")

    # Remover nulos nas colunas essenciais
    cols_obrigatorias = ["ageofonset", "survivalmonths", "progressionrate", "sex", "gene"]
    # Verifica quais colunas existem antes de tentar limpar
    cols_existentes = [c for c in cols_obrigatorias if c in df.columns]
    df_clean = df.dropna(subset=cols_existentes)

    # Indexar colunas de texto (String -> Número)
    indexers = []
    # Nota: Se na camada trusted 'sex' ou 'gene' já forem números, o StringIndexer pode dar erro.
    # Vou assumir que ainda são texto (ex: "Male", "SOD1"). Se já forem números, avise!
    cols_categoricas = ["sex", "gene", "onsetsite", "familyhistory", "progressionrate"]
    
    for c in cols_categoricas:
        if c in df_clean.columns:
            # Se a coluna já for numérica, não indexamos, apenas renomeamos ou usamos direto
            # Mas como padrão ML, vamos garantir o indexer se for string
            dtype = dict(df_clean.dtypes)[c]
            if dtype == 'string':
                output_name = f"{c}_index" if c != "progressionrate" else "label_progression"
                indexers.append(StringIndexer(inputCol=c, outputCol=output_name))
            else:
                # Se já for numero (int/double), apenas criamos um alias se for o label
                if c == "progressionrate":
                    df_clean = df_clean.withColumn("label_progression", col(c))

    pipeline_preprocessing = Pipeline(stages=indexers)
    model_preprocessing = pipeline_preprocessing.fit(df_clean)
    df_transformed = model_preprocessing.transform(df_clean)

    # --- 4. ALGORITMO 1: CLUSTERIZAÇÃO (K-MEANS) ---
    print("Treinando K-Means...")
    
    vec_assembler_cluster = VectorAssembler(inputCols=["ageofonset", "survivalmonths"], outputCol="features_cluster")
    df_cluster_ready = vec_assembler_cluster.transform(df_transformed)

    kmeans = KMeans(featuresCol="features_cluster", predictionCol="cluster_id", k=3, seed=1)
    model_kmeans = kmeans.fit(df_cluster_ready)
    predictions_cluster = model_kmeans.transform(df_cluster_ready)
    
    evaluator_cl = ClusteringEvaluator(featuresCol="features_cluster", predictionCol="cluster_id")
    silhouette = evaluator_cl.evaluate(predictions_cluster)
    print(f"Métrica Silhouette do Cluster: {silhouette:.4f}")

    # --- 5. ALGORITMO 2: CLASSIFICAÇÃO (RANDOM FOREST) ---
    print("Treinando Random Forest...")

    cols_classificacao = [
        "ageofonset", "cluster_id", "sex_index", 
        "gene_index", "familyhistory_index", 
        "onsetsite_index", "motorsymptomsscore"
    ]
    # Filtra apenas colunas que realmente existem no dataframe final
    cols_finais = [c for c in cols_classificacao if c in predictions_cluster.columns]
    
    vec_assembler_class = VectorAssembler(inputCols=cols_finais, outputCol="features_class")
    df_class_ready = vec_assembler_class.transform(predictions_cluster)

    train, test = df_class_ready.randomSplit([0.8, 0.2], seed=42)

    rf = RandomForestClassifier(
        labelCol="label_progression", 
        featuresCol="features_class", 
        predictionCol="predicted_class_raw",
        numTrees=50, 
        maxDepth=10
    )
    
    model_rf = rf.fit(train)
    df_final_predictions = model_rf.transform(df_class_ready)

    accuracy = MulticlassClassificationEvaluator(
        labelCol="label_progression", predictionCol="predicted_class_raw", metricName="accuracy"
    ).evaluate(df_final_predictions)
    print(f"Acurácia do modelo RF: {accuracy*100:.2f}%")

    # --- 6. FORMATAÇÃO FINAL E SALVAMENTO ---
    print("Formatando dataset final...")

    df_enriched = df_final_predictions \
        .withColumnRenamed("predicted_class_raw", "predicted_risk_group") \
        .drop("features_cluster", "features_class", "features_raw")
    
    df_export = df_enriched.drop("features") 

    print(f"Salvando dataset enriquecido em: {caminho_saida}")
    
    df_export.write.mode("overwrite").parquet(caminho_saida)

    print("Job concluído com sucesso!")
    job.commit()

except Exception as e:
    print(f"ERRO CRÍTICO NO JOB: {str(e)}")
    sys.exit(1)
