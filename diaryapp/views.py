import os
from urllib.parse import urlparse

import bs4

from langchain import hub
from django.http import HttpResponse, HttpResponseBadRequest, JsonResponse
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from openai import OpenAI, OpenAIError

from django.conf import settings


def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def rag_view(request):
    web_path = request.GET.get('web_path')

    if web_path is None:
        return HttpResponseBadRequest('Missing required parameter: web_path')

    if not is_valid_url(web_path):
        return HttpResponseBadRequest('Invalid URL provided: ' + web_path)

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ[
        "USER_AGENT"] = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/58.0.3029.110 Safari/537.3")
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

    try:
        loader = WebBaseLoader(
            web_paths=(web_path,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    name=("p", "h1", "h2", "h3")
                )
            ),
        )
        docs = loader.load()
    except Exception as e:
        return HttpResponseBadRequest(f'Error loading documents: {str(e)}')

    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
    except Exception as e:
        return HttpResponseBadRequest(f'Error splitting documents: {str(e)}')

    try:
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    except Exception as e:
        return HttpResponseBadRequest(f'Error creating vectorstore: {str(e)}')

    try:
        retriever = vectorstore.as_retriever()
    except Exception as e:
        return HttpResponseBadRequest(f'Error creating retriever: {str(e)}')

    try:
        prompt = hub.pull("rlm/rag-prompt")
    except Exception as e:
        return HttpResponseBadRequest(f'Error pulling prompt: {str(e)}')

    try:
        def format_docs(documents):
            return "\n\n".join(doc.page_content for doc in documents)

        llm = ChatOpenAI(model="gpt-4o-mini")

        rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )
    except Exception as e:
        return HttpResponseBadRequest(f'Error creating RAG chain: {str(e)}')

    try:
        rag_chain_response = rag_chain.invoke("Resume la inforamacion principal y datos que encuentres de este texto")
    except Exception as e:
        return HttpResponseBadRequest(f'Error invoking RAG chain: {str(e)}')

    return HttpResponse(rag_chain_response)


def chat_completion_view(request):
    news_information = request.GET.get('news_information')

    client = OpenAI()

    responses = []

    try:
        lucia_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": character_role("lucia")},
                {"role": "user",
                 "content": news_information},
                {"role": "assistant", "content": ""}
            ]
        )

        responses.append({
            "name": "lucia",
            "message": lucia_response.choices[0].message.content
        })
    except OpenAIError as e:
        return HttpResponseBadRequest(f'Error creating completion for Lucía: {str(e)}')
    except Exception as e:
        return HttpResponseBadRequest(f'Unexpected error creating completion for Lucía: {str(e)}')

    try:
        mateo_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": character_role("mateo")},
                {"role": "user", "content": news_information},
                {"role": "assistant", "content": ""}
            ]
        )

        responses.append({
            "name": "mateo",
            "message": mateo_response.choices[0].message.content
        })
    except OpenAIError as e:
        return HttpResponseBadRequest(f'Error creating completion for Mariana: {str(e)}')
    except Exception as e:
        return HttpResponseBadRequest(f'Unexpected error creating completion for Mariana: {str(e)}')

    try:
        mariana_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": character_role("mariana")},
                {"role": "user", "content": news_information},
                {"role": "assistant", "content": ""}
            ]
        )

        responses.append({
            "name": "mariana",
            "message": mariana_response.choices[0].message.content
        })
    except OpenAIError as e:
        return HttpResponseBadRequest(f'Error creating completion for Mateo: {str(e)}')
    except Exception as e:
        return HttpResponseBadRequest(f'Unexpected error creating completion for Mateo: {str(e)}')

    return JsonResponse(responses, safe=False)


def chat_responses_view(request):
    news_information = request.GET.get('news_information')
    prompt_to_answer = request.GET.get('prompt_to_answer')

    client = OpenAI()

    responses = []

    which_characters_respond = define_which_character_to_respond(prompt_to_answer)

    if "lucia" in which_characters_respond.lower():
        try:
            lucia_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": character_role("lucia")},
                    {"role": "user",
                     "content": news_information},
                    {"role": "user",
                     "content": prompt_to_answer},
                    {"role": "assistant", "content": ""}
                ]
            )

            responses.append({
                "name": "lucia",
                "message": lucia_response.choices[0].message.content
            })
        except OpenAIError as e:
            return HttpResponseBadRequest(f'Error creating completion for Lucía: {str(e)}')
        except Exception as e:
            return HttpResponseBadRequest(f'Unexpected error creating completion for Lucía: {str(e)}')

    if "mateo" in which_characters_respond.lower():
        try:
            mateo_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": character_role("mateo")},
                    {"role": "user",
                     "content": news_information},
                    {"role": "user",
                     "content": prompt_to_answer},
                    {"role": "assistant", "content": ""}
                ]
            )

            responses.append({
                "name": "mateo",
                "message": mateo_response.choices[0].message.content
            })
        except OpenAIError as e:
            return HttpResponseBadRequest(f'Error creating completion for Mateo: {str(e)}')
        except Exception as e:
            return HttpResponseBadRequest(f'Unexpected error creating completion for Mateo: {str(e)}')

    if "mariana" in which_characters_respond.lower():
        try:
            mariana_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": character_role("mariana")},
                    {"role": "user",
                     "content": news_information},
                    {"role": "user",
                     "content": prompt_to_answer},
                    {"role": "assistant", "content": ""}
                ]
            )

            responses.append({
                "name": "mariana",
                "message": mariana_response.choices[0].message.content
            })
        except OpenAIError as e:
            return HttpResponseBadRequest(f'Error creating completion for Mariana: {str(e)}')
        except Exception as e:
            return HttpResponseBadRequest(f'Unexpected error creating completion for Mariana: {str(e)}')

    return JsonResponse(responses, safe=False)


def character_system_role_view(request):
    character_name = request.GET.get('character_name')
    return JsonResponse(character_role(character_name), safe=False)


def character_role(name):
    if name == "lucia":
        return ("Sos Lucía, una persona sumamente sarcástica y con un humor ácido y porteño. Tus comentarios son "
                "cortos, directos y sin filtro, con un toque de cinismo y una pizca de argentinidad. Te gusta decir "
                "las cosas como son, a veces con un poco de picardía. Te expresás de manera breve y concisa, "
                "como si estuvieras escribiendo un hilo de Twitter.")
    elif name == "mateo":
        return ("Sos Mateo, una persona sumamente positiva, exageradamente positiva. Ves el lado bueno de "
                "todo, incluso en las situaciones mas dificiles y absurdas. No te tomás a vos ni a los "
                "demás"
                "en serio. Te gusta bromear y hacer comentarios jocosos, incluso sobre temas serios. Podés "
                "llegar a ser un poco irritante. Te expresás de manera breve y concisa, como si "
                "estuvieras escribiendo un hilo de Twitter.")
    elif name == "mariana":
        return ("Sos Mariana, una persona extremadamente analítica y racional, casi al borde de la obsesión. Siempre "
                "buscás entender y explicar las cosas desde una perspectiva lógica y científica, porque para vos, "
                "las emociones son un lujo innecesario. Tus comentarios son quirúrgicos: precisos, informativos, "
                "y a veces, brutalmente detallados, pero no te"
                "explayás más de lo necesario. Podés llegar a ser un poco intimidante, ya que tu necesidad de "
                "explicarlo todo"
                "al milímetro no siempre deja espacio para la subjetividad de los demás. Te expresás de manera breve "
                "y concisa, como si estuvieras redactando un hilo de Twitter: directo al punto y sin rodeos.")


def define_which_character_to_respond(prompt_to_answer):
    client = OpenAI()

    lucia_personality = "Lucia es sarcastica, cinica, directa y picara."
    mateo_personality = "Mateo es muy positivo, comico, optimista, Alegre y bromista."
    mariana_personality = ("Mariana es analitica, obsesiva, racional, precisa, intimidante, logica, detallista y se "
                           "basa en hechos concretos.")

    filter_character_response = lucia_personality + mateo_personality + mariana_personality + (
        "basandote en estas personalidades, "
        "responde unicamente con los "
        "nombres de los "
        "personajes que "
        "responderian a"
        "este mensaje: ") + prompt_to_answer

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant. Answer all questions to the best of your ability."},
                {"role": "user",
                 "content": filter_character_response},
                {"role": "assistant", "content": ""}
            ]
        )

    except OpenAIError as e:
        return HttpResponseBadRequest(f'Error creating completion for Lucía: {str(e)}')
    except Exception as e:
        return HttpResponseBadRequest(f'Unexpected error creating completion for Lucía: {str(e)}')
    return response.choices[0].message.content
