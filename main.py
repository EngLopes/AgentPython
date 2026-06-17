
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM # <-- Nova importação do Ollama!
app = FastAPI(title="API de Suporte de TI - RAG")
CAMINHO_DB = './db'

prompt_template = """
  Você é um assistente técnico rigoroso e objetivo.
  Responda à pergunta do usuário usando EXCLUSIVAMENTE as informações fornecidas no contexto abaixo.

  REGRAS:
  1. Se o contexto não contiver a resposta exata, diga APENAS: "A documentação não informa os valores exatos."
  2. NÃO invente informações e NÃO use seu conhecimento prévio.
  3. Seja direto e curto. Não adicione informações extras que não foram perguntadas.

  Responda a pergunta do usuário:
  {pergunta}

  Contexto (informações extraídas da documentação):
  {informacoes}

 
"""

pergunta = input("Digite sua pergunta: ")

# 1. Carrega a parte de Busca (O Bibliotecário)
funcao_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db = Chroma(persist_directory=CAMINHO_DB, embedding_function=funcao_embeddings)

def perguntar():
    # Faz a busca no banco
    resultados = db.similarity_search_with_relevance_scores(pergunta, k=3)

    if len(resultados) == 0 or resultados[0][1] < 0.2:
        print("Desculpe, não tenho informações suficientes para responder a sua pergunta.")
        return
    else:
        texto_resultado = []
        for doc, score in resultados:
            texto_resultado.append(doc.page_content)

        base_conhecimento = "\n\n-----\n\n".join(texto_resultado)
        

    # Prepara o texto final
    prompt = PromptTemplate.from_template(prompt_template)
    prompt_formatado = prompt.invoke({"pergunta": pergunta, "informacoes": base_conhecimento})

    print("\n⏳ Pensando e gerando a resposta...\n")

    # Instancia o LLM do Ollama
    llm = OllamaLLM(
        model="qwen2.5:0.5b", 
        num_gpu=0,
        temperature=0.3,
        num_thread=4
    )

    # Faz a IA local gerar uma resposta amigável e útil
    resposta_final = llm.invoke(prompt_formatado.text)
    
    print("🤖 RESPOSTA DA IA:")
    print("--------------------------------------------------")
    print(resposta_final)
    print("--------------------------------------------------")

perguntar()