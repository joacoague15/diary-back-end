import os

import bs4

from langchain import hub
from django.http import HttpResponse
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from openai import OpenAI

from django.conf import settings


#link is hardcoded for now
def rag_view(request):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ[
        "USER_AGENT"] = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/58.0.3029.110 Safari/537.3")
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

    loader = WebBaseLoader(
        web_paths=("https://www.clarin.com/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                name=("p", "h1", "h2", "h3")
            )
        ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(documents):
        return "\n\n".join(doc.page_content for doc in documents)

    llm = ChatOpenAI(model="gpt-4o-mini")

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    rag_chain_response = rag_chain.invoke(
        "Puedes darme los principales titulos y novedades que aparecen en la portada de este diario?")

    return HttpResponse(rag_chain_response)


# rag chain response is hardcoded for now
def chat_completion_view(request):
    client = OpenAI()

    responses = []

    lucia_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "Sos Lucía, una persona sumamente sarcástica y con un humor ácido y porteño. Tus comentarios "
                        "son cortos, directos y sin filtro, con un toque de cinismo y una pizca de argentinidad. Te "
                        "gusta decir las cosas como son, a veces con un poco de picardía."},
            {"role": "user", "content": "Venezuela anda en un conflicto muy grande entre su presidente y el pueblo."},
            {"role": "assistant", "content": ""}
        ]
    )

    responses.append(lucia_response.choices[0].message.content)

    mateo_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "Sos Mateo, una persona sumamente positiva, exageradamente positiva. Ves el lado bueno de "
                        "todo, incluso en las situaciones mas dificiles y absurdas. No te tomás a vos ni a los demás "
                        "en serio. Te gusta bromear y hacer comentarios jocosos, incluso sobre temas serios. Podés "
                        "llegar a ser un poco irritante."},
            {"role": "user", "content": "Venezuela anda en un conflicto muy grande entre su presidente y el pueblo."},
            {"role": "assistant", "content": ""}
        ]
    )

    responses.append(mateo_response.choices[0].message.content)

    return HttpResponse(responses)

