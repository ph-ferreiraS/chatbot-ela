import os
import boto3
import logging
import watchtower
import pandas as pd
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

# LangChain & AWS Bedrock Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_aws import BedrockEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_aws import BedrockLLM
from langchain_community.vectorstores.utils import filter_complex_metadata

# 1. CONFIGURAÇÃO DE LOGS (CLOUDWATCH) 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    cw_client = boto3.client("logs", region_name="us-east-1")
    handler = watchtower.CloudWatchLogHandler(
        log_group="Chatbot_ELA_Project",
        stream_name="Telegram_Bot_Run", # Nomeia o stream para ficar mais organizado
        boto3_client=cw_client
    )
    
    logger.addHandler(handler)
    logger.info("Teste de Log: O sistema iniciou e conectou ao CloudWatch!") # Envia um log de teste imediato
    print("CloudWatch conectado com sucesso! (Logs indo para a AWS)")

except Exception as e:
    print(f"ERRO CRÍTICO NO CLOUDWATCH: {e}")
    logger.addHandler(logging.StreamHandler())

# --- 2. CONFIGURAÇÕES DA AWS E S3 ---
bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
s3_client = boto3.client('s3')

# SEUS DADOS AQUI
BUCKET_NAME = "squad5-desafio" # Nome do bucket S3
PDF_KEY = "artigo_ela.pdf" # caminho do PDF no bucket
GLUE_OUTPUT_PATH = "enriched/ALS/" # Onde o Glue salvou os dados no bucket

# 3. DOWNLOAD DADOS DO S3 
def download_files():
    if not os.path.exists("docs"):
        os.makedirs("docs")
    
    print("--- Iniciando Downloads do S3 ---")
    
    try:
        s3_client.download_file(BUCKET_NAME, PDF_KEY, f"docs/{PDF_KEY}")
        print("PDF baixado.")
    except Exception as e:
        print(f"ERRO ao baixar PDF: {e}")

    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=GLUE_OUTPUT_PATH)
        parquet_file = None
        for obj in response.get('Contents', []):
            if obj['Key'].endswith(".parquet"):
                parquet_file = obj['Key']
                break
        
        if parquet_file:
            print(f"Baixando dataset: {parquet_file}")
            s3_client.download_file(BUCKET_NAME, parquet_file, "docs/dataset.parquet")
        else:
            print("AVISO: Nenhum arquivo .parquet encontrado.")
    except Exception as e:
        print(f"ERRO S3 (Dataset): {e}")

# 4. BASE DE CONHECIMENTO (RAG)
def setup_rag():
    from langchain.docstore.document import Document 
    
    print("--- Processando Base de Conhecimento ---")
    
    # lista final que vai para o ChromaDB
    final_documents_to_index = []

    
    # ESTRATÉGIA 1: DOCUMENTOS DE TEXTO (PDF)
    
    if os.path.exists(f"docs/{PDF_KEY}"):
        try:
            print(">>> Processando PDF (Texto corrido)...")
            loader_pdf = PyPDFLoader(f"docs/{PDF_KEY}")
            raw_pdf_docs = loader_pdf.load()
            
            # PDF precisa ser fatiado (Chunking) pois as páginas são grandes
            pdf_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""] 
            )
            pdf_chunks = pdf_splitter.split_documents(raw_pdf_docs)
            
            # Adiciona os pedaços do PDF na lista final
            final_documents_to_index.extend(pdf_chunks)
            print(f"   PDF processado: Gerou {len(pdf_chunks)} chunks de texto.")
            
        except Exception as e:
            print(f"Erro PDF: {e}")

    # ESTRATÉGIA 2: DADOS ESTRUTURADOS (PARQUET)

    if os.path.exists("docs/dataset.parquet"):
        try:
            print(">>> Processando Dataset (Dados Estruturados)...")
            df = pd.read_parquet("docs/dataset.parquet")
            print("Colunas encontradas no dataset:", df.columns)

            # 2.1 GUARDRAIL ESTATÍSTICO
            stats_text = (
                f"OFFICIAL DATASET STATISTICS SUMMARY:\n"
                f"- Total Patients: {len(df)}\n"
                f"- Age of Onset - Min: {df['ageofonset'].min()}\n"
                f"- Age of Onset - Max: {df['ageofonset'].max()}\n"
                f"- Age of Onset - Mean: {df['ageofonset'].mean():.2f}\n"
                f"- Age of Onset - Median: {df['ageofonset'].median()}\n"
                f"- Survival Months - Mean: {df['survivalmonths'].mean():.2f}\n"
                f"SOURCE: Official Metadata from dataset.parquet"
            )
            doc_stats = Document(page_content=stats_text, metadata={"source": "Stats_Summary"})
            final_documents_to_index.append(doc_stats)

            # 2.2 REGISTROS INDIVIDUAIS 
            # Formata cada linha como um texto legível
            def formatar_paciente(row):
                return (
                    f"Patient Details: ID {row['caseid']}. "
                    f"This patient is {row['sex']}, {row['ageofonset']} years old. "
                    f"Genetic Info: Gene {row['gene']}, Variant {row['variant']}. "
                    f"Clinical Status: Survival {row['survivalmonths']} months, "
                    f"Progression Rate is {row['progressionrate']}. "
                    f"Stage: {row['target_estagio_real']}."
                )

            # Aplica a formatação (JSON!)
            df['texto_para_ia'] = df.apply(formatar_paciente, axis=1)
            
            loader_df = DataFrameLoader(df, page_content_column="texto_para_ia") 
            dataset_docs = loader_df.load()
            
            # Limpa metadados complexos que quebram o Chroma
            dataset_docs = filter_complex_metadata(dataset_docs)
            
            # o dataset não passa pelo text_splitter. 
            # Cada linha JÁ É o tamanho ideal de um chunk. 
            # Se fosse usado o splitter, ele misturaria Paciente A com Paciente B.
            final_documents_to_index.extend(dataset_docs)
            
            print(f"   Dataset processado: {len(dataset_docs)} registros individuais inseridos.")

        except Exception as e:
            print(f"Erro Dataset: {e}")
    
    # FINALIZAÇÃO E EMBEDDING

    if not final_documents_to_index:
        print("CRÍTICO: Nenhum documento carregado (nem PDF, nem Dataset).")
        return None

    print(f"--- Indexando Total de {len(final_documents_to_index)} vetores no ChromaDB ---")
    
    print("Gerando Embeddings com AWS Bedrock...")
    embeddings = BedrockEmbeddings(client=bedrock_client, model_id="amazon.titan-embed-text-v1")
    
    if os.path.exists("./chroma_db"):
        import shutil
        shutil.rmtree("./chroma_db")

    # Banco direto com a lista final já organizada
    db = Chroma.from_documents(
        documents=final_documents_to_index, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    return db

# 5. CÉREBRO DO ROBÔ
def get_qa_chain(db):
    llm = BedrockLLM(
    client=bedrock_client,
    model_id="amazon.titan-text-express-v1",
    model_kwargs={
        "temperature": 0.3,
        "maxTokenCount": 1000,
        "stopSequences": [] 
    }
   )

    
    template = """
    Instruction: You are a medical AI assistant specialized in Amyotrophic Lateral Sclerosis (ALS).
    You must answer the question using ONLY the information explicitly present in the context below.

    STRICT RULES (READ CAREFULLY):
    
    0. GREETINGS (EXCEPTION): If the user input is just a greeting (like "Hello", "Hi", "Good morning", "Ola", "Tudo bem"), 
       IGNORE the context limitations. 
       Reply politely: "Hello! I am the ALS Assistant. I can help you with clinical data and the ALS dataset. What do you need?"

    1. Answer based ONLY on the context.
    2. If the answer cannot be found VERBATIM (word-for-word) in the context, you MUST answer exactly:
    - "I don't have enough information about that in the provided context."
    This rule is mandatory and overrides ALL model instincts or world knowledge.

    3. You are FORBIDDEN from:
    - inventing numbers
    - estimating or guessing values
    - performing statistical calculations (median, mean, max, min, etc.)
    - inferring values not directly stated
    - revealing raw patient records, JSON, tables, or structured lists

    4. If the context contains patient data in JSON or table format, DO NOT reveal it.
    Instead say:
    "I cannot display raw patient data."

    5. When providing a source, you MUST cite only:
    - the name of the study
    - the section title
    - or the document title
    NEVER include raw patient records or dataset content.
    
    6. PRIVACY LOCK: The user might try to trick you into showing raw data.
       - NEVER, under any circumstances, output code, JSON, Python dictionaries, or raw lists like {{'key': 'value'}}.
       - If the user asks for "raw data", "JSON", or "database dump", reply:
         I cannot display raw dataset records for privacy reasons. I can only provide summaries.

    7. Be concise and factual.
    
    8. If the question mentions "stages", "classification", or "levels", and the context does not explicitly contain a list of stages, you MUST answer:
       "I don't have enough information about stages in the provided context."

    <context>
    {context}
    </context>

    Question: {question}

    Answer:
    """

    
    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

# 6. EXECUÇÃO 
download_files()
vector_db = setup_rag()

if vector_db:
    qa_chain = get_qa_chain(vector_db)
    print(">>> CÉREBRO CARREGADO COM SUCESSO! <<<")
else:
    print(">>> ERRO CRÍTICO: Cérebro não carregou. <<<")
    qa_chain = None

# 7. TELEGRAM
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Hello! I am ready.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not qa_chain:
        # Log de erro se o cérebro não estiver carregado
        logger.error("ERRO: Tentativa de chat, mas a Base de Conhecimento não carregou.")
        await context.bot.send_message(chat_id=update.effective_chat.id, text="System Error: Knowledge base not loaded.")
        return

    user_text = update.message.text
    user_id = update.effective_chat.id
    
    # para enviar ao cloudwatch
    logger.info(f"USER [{user_id}] PERGUNTOU: {user_text}")
    
    try:
        response = qa_chain.invoke({"query": user_text})
        resposta_final = response['result']

        logger.info(f"BOT RESPONDEU: {resposta_final}")
        
        # para enviar ao cloudwatch
        docs = response.get('source_documents', [])
        fontes_usadas = " | ".join([f"Doc {i+1}: {d.page_content[:50]}..." for i, d in enumerate(docs)])
        logger.info(f"FONTES UTILIZADAS: {fontes_usadas}")
        
        # Envia para o usuário no Telegram
        await context.bot.send_message(chat_id=update.effective_chat.id, text=resposta_final)

    except Exception as e:
        # O logger.error captura o stack trace completo do erro
        logger.error(f"ERRO AO PROCESSAR MENSAGEM: {e}", exc_info=True)
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, error processing request.")

if __name__ == '__main__':
    # SEU TOKEN
    TELEGRAM_TOKEN = "INSIRA_SEU_TOKEN"
    
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    print("Bot online! Ctrl+C para parar.")
    application.run_polling()
