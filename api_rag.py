from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# 1. Cria a API
app = FastAPI(title="API de Suporte de TI - RAG")

print("⏳ Ligando os motores da IA... Aguarde.")

# 2. Carrega o Banco de Dados (Uma vez só)
CAMINHO_DB = './db'
funcao_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory=CAMINHO_DB, embedding_function=funcao_embeddings)

# 3. Carrega o modelo do Ollama (Uma vez só, já configurado para CPU)
llm = OllamaLLM(
    model="qwen2.5:0.5b", 
    temperature=0.3,
    num_gpu=0,
    num_thread=6,
    num_predict=300

)

# 4. O Prompt Restritivo
prompt_template = """
Você é um atendente de vendas para uma loja de cabelos/laces chamado Markete.
Responda à pergunta do usuário usando EXCLUSIVAMENTE as informações fornecidas no contexto abaixo.

REGRAS:
1. Se o contexto não contiver a resposta, diga APENAS: "A documentação não informa."
2. NÃO invente informações.
3. Seja direto e curto.

Contexto:
{informacoes}

Pergunta: {pergunta}

Resposta:
"""

# 5. Define como os dados devem chegar na API (JSON)
class Requisicao(BaseModel):
    pergunta: str

# 6. A Rota da API (Onde a mágica acontece)
@app.post("/atendimento")
async def atendimento_bot(requisicao: Requisicao):
    # Faz a busca no banco vetorial
    resultados = db.similarity_search_with_relevance_scores(requisicao.pergunta, k=3)

    # Se não achar nada bom o suficiente, barra aqui mesmo
    if len(resultados) == 0 or resultados[0][1] < 0.2:
        return {
            "pergunta": requisicao.pergunta,
            "resposta": "Desculpe, não encontrei nenhuma documentação sobre esse problema.",
            "status": "sem_contexto"
        }

    # Junta os textos encontrados
    textos = [doc.page_content for doc, score in resultados]
    base_conhecimento = "\n\n-----\n\n".join(textos)

    # Prepara a pergunta para a IA
    prompt = PromptTemplate.from_template(prompt_template)
    prompt_formatado = prompt.invoke({"pergunta": requisicao.pergunta, "informacoes": base_conhecimento})

    # Pede a resposta para o Ollama
    resposta_ia = llm.invoke(prompt_formatado.text)

    # Devolve a resposta estruturada em JSON
    return {
        "pergunta": requisicao.pergunta,
        "resposta": resposta_ia.strip(),
        "documentos_consultados": len(resultados),
        "status": "sucesso"
    }