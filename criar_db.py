from langchain_community.document_loaders import DirectoryLoader, PyPDFDirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
PASTA_BASE = 'base'

def criar_db():
  documentos_pdf = carregar_documentos_pdf()
  documentos_txt = carregar_documentos_txt()
  documentos = documentos_pdf + documentos_txt
  chunks = dividir_em_chuncks(documentos)
  vetorizar_chuncks(chunks)

def carregar_documentos_pdf():
  carregador = PyPDFDirectoryLoader(PASTA_BASE, glob="**/*.pdf")
  documentos = carregador.load()
  return documentos

def carregar_documentos_txt():
  carregador = DirectoryLoader(
    PASTA_BASE,
    glob="**/*.txt",
    show_progress=True,
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
  )
  documentos = carregador.load()
  return documentos

def dividir_em_chuncks(documentos):
  separador_documentos = RecursiveCharacterTextSplitter(
    chunk_size=1000, #tamanho do chunk (quantidade de caracteres)
    chunk_overlap=200, #quantidade de caracteres que se sobrepõe entre os chunks
    length_function=len, #função para calcular o tamanho do chunk (número de caracteres)
    add_start_index=True #informar o indice de onde iniciou e acabou a chunk
  )

  chunks = separador_documentos.split_documents(documentos)
  print(len(chunks))
  return chunks

def vetorizar_chuncks(chunks):
  db = Chroma.from_documents(chunks, embeddings, persist_directory="db")
  print("Banco de dados criado com sucesso!")
criar_db()