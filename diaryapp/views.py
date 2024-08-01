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

from django.conf import settings


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


def about_view(request):
    return HttpResponse("This is the About Page.")


def contact_view(request):
    return HttpResponse("Contact us at contact@example.com.")
