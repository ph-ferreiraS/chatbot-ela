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

# --- 1. CONFIGURAÇÃO DE LOGS (CLOUDWATCH) ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Correção: Forçar a região us-east-1 para o CloudWatch não se perder
    boto3.setup_default_session(region_name='us-east-1')
    handler = watchtower.CloudWatchLogHandler(log_group="Chatbot_ELA_Project")
    logger.addHandler(handler)
    print("CloudWatch conectado com sucesso! (Logs indo para a AWS)")
except Exception as e:
    print(f"ERRO CRÍTICO NO CLOUDWATCH: {e}")

# --- 2. CONFIGURAÇÕES DA AWS E S3 ---
bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
s3_client = boto3.client('s3')

# SEUS DADOS AQUI
BUCKET_NAME = "ela-datalake"
PDF_KEY = "artigo_ela.pdf"
GLUE_OUTPUT_PATH = "ml/enriched/dataset_enriquecido/"

# --- 3. DOWNLOAD DADOS DO S3 ---
def download_files():
    if not os.path.exists("docs"):
        os.makedirs("docs")
    
    print("--- Iniciando Downloads do S3 ---")
    
    # Baixar PDF
    try:
        s3_client.download_file(BUCKET_NAME, PDF_KEY, f"docs/{PDF_KEY}")
        print("PDF baixado.")
    except Exception as e:
        print(f"ERRO ao baixar PDF: {e}")

    # Baixar Dataset
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

# --- 4. PREPARAR BASE DE CONHECIMENTO (RAG) ---
def setup_rag():
    print("--- Processando Base de Conhecimento ---")
    all_docs = []

    # A. Carregar PDF
    if os.path.exists(f"docs/{PDF_KEY}"):
        try:
            loader_pdf = PyPDFLoader(f"docs/{PDF_KEY}")
            docs_pdf = loader_pdf.load()
            all_docs.extend(docs_pdf)
            print(f"PDF Carregado: {len(docs_pdf)} páginas.")
        except Exception as e:
            print(f"Erro PDF: {e}")

    # B. Carregar Parquet
    if os.path.exists("docs/dataset.parquet"):
        try:
            df = pd.read_parquet("docs/dataset.parquet")
            # Converter para texto
            df['texto_para_ia'] = df.apply(lambda row: f"Patient Record: {row.to_dict()}", axis=1)
            
            loader_df = DataFrameLoader(df, page_content_column="texto_para_ia") 
            docs_df = loader_df.load()
            
            # FILTRO DE METADADOS (Importante!)
            docs_df = filter_complex_metadata(docs_df)
            
            all_docs.extend(docs_df)
            print(f"Dataset Carregado: {len(docs_df)} registros.")
        except Exception as e:
            print(f"Erro Dataset: {e}")
    
    if not all_docs:
        print("CRÍTICO: Nenhum documento carregado.")
        return None

    # Chunking e Indexação
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_docs)
    
    print("Gerando Embeddings com AWS Bedrock...")
    embeddings = BedrockEmbeddings(client=bedrock_client, model_id="amazon.titan-embed-text-v1")
    
    # Limpeza preventiva do banco
    if os.path.exists("./chroma_db"):
        import shutil
        shutil.rmtree("./chroma_db")

    db = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="./chroma_db")
    return db

# --- 5. CÉREBRO DO ROBÔ (VERSÃO ESTÁVEL + PROMPT INTELIGENTE) ---
def get_qa_chain(db):
    # Voltamos para a configuração técnica que funcionava (sem stopSequences)
    llm = BedrockLLM(
        client=bedrock_client,
        model_id="amazon.titan-text-express-v1",
        model_kwargs={
            "temperature": 0.1, # Baixa criatividade (quase zero)
            "maxTokenCount": 512
        }
    )
    
    # Mantemos o Prompt "Blindado" para evitar alucinações via texto
    template = """
    Instruction: You are a medical AI assistant specialized in Amyotrophic Lateral Sclerosis (ALS).
    Your task is to answer the question based ONLY on the provided context.
    
    Rules:
    1. If the exact answer is not in the context, say "I don't have enough information."
    2. Do NOT guess numbers.
    3. Do NOT calculate averages or statistics if they are not explicitly written in the text.
    4. Answer directly and concisely.
    
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

# --- 6. EXECUÇÃO ---
download_files()
vector_db = setup_rag()

if vector_db:
    qa_chain = get_qa_chain(vector_db)
    print(">>> CÉREBRO CARREGADO COM SUCESSO! <<<")
else:
    print(">>> ERRO CRÍTICO: Cérebro não carregou. <<<")
    qa_chain = None

# --- 7. TELEGRAM ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Hello! I am ready.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not qa_chain:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="System Error: Knowledge base not loaded.")
        return

    user_text = update.message.text
    print(f"\nPergunta recebida: {user_text}")
    
    try:
        response = qa_chain.invoke({"query": user_text})
        
        # DEBUG NO TERMINAL
        print("--- CONTEXTO USADO ---")
        docs = response.get('source_documents', [])
        for i, doc in enumerate(docs):
            print(f"[{i+1}] {doc.page_content[:100].replace(chr(10), ' ')}...")
        print("----------------------")
        
        await context.bot.send_message(chat_id=update.effective_chat.id, text=response['result'])

    except Exception as e:
        logger.error(f"Erro ao processar: {e}")
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, error processing request.")

if __name__ == '__main__':
    # SEU TOKEN
    TELEGRAM_TOKEN = "API-TELEGRAM-TOKEN"
    
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    print("Bot online! Ctrl+C para parar.")
    application.run_polling()
