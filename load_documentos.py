import re
import os
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader, YoutubeLoader, CSVLoader,PyPDFLoader, TextLoader
from youtube_transcript_api import YouTubeTranscriptApi
from fake_useragent import UserAgent
from time import sleep

def carregar_site(url):
    documento = ''
    for i in range(5):
        try:
            os.environ['USER_AGENT'] = UserAgent().random
            loader = WebBaseLoader(url, raise_for_status=True)
            lista_documentos = loader.load()
            documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
            break
        except:
            print(f'Erro ao carregar o site {i+1}')
            sleep(3)
    if documento == '':
        st.error('Não foi possível carregar o site.')
        st.stop()
    return documento

def carregar_youtube(url):
    video_id = None
    match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
    if match:
        video_id = match.group(1)
    else:
        st.error('Não foi possível extrair o ID do vídeo da URL.')
        st.stop()
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript(['pt'])
        entries = transcript.fetch()
        documento = "\n\n".join(entry.text for entry in entries)
        return documento
    except Exception as e:
        st.error(f"Erro ao carregar a transcrição do vídeo: {str(e)}")
        st.stop()

def carregar_csv(caminho):
    loader = CSVLoader(caminho)
    lista_documentos = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return documento

def carregar_pdf(caminho):
    loader = PyPDFLoader(caminho)
    lista_documentos = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return documento

def carregar_txt(caminho):
    loader = TextLoader(caminho)
    lista_documentos = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return documento

