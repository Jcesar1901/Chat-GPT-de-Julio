import tempfile
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from load_documentos import *
from langchain.prompts import ChatPromptTemplate

provedores = {'Groq': {'modelos': ['llama-3.1-70b-versatile', 'gemma2-9b-it', 'mixtral-8x7b-32768'], 'chat': ChatGroq},
              'OpenAI': {'modelos': ['gpt-4o-mini', 'gpt-4o', 'o1-mini'], 'chat': ChatOpenAI}}

arquivos_validos = [
    'Youtube', 'Site', 'Pdf', 'Csv', 'Txt'
]

memoria = ConversationBufferMemory()
memoria.chat_memory.add_ai_message('Aoba, você está falando com o Chat de Julio! Faça uma pergunta.')

def carregar_arquivos(tipo_arquivo, arquivo):
    if tipo_arquivo == 'Site':
        documento = carregar_site(arquivo)
    if tipo_arquivo == 'Youtube':
        documento = carregar_youtube(arquivo)
    if tipo_arquivo == 'Pdf':
        with tempfile.NamedTemporaryFile(suffix = '.pdf', delete = False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carregar_pdf(nome_temp)
    if tipo_arquivo == 'Csv':
        with tempfile.NamedTemporaryFile(suffix = '.csv', delete = False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carregar_csv(nome_temp)
    if tipo_arquivo == 'Txt':
        with tempfile.NamedTemporaryFile(suffix = '.txt', delete = False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carregar_txt(nome_temp)

    return documento

def carregar_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo):
    
    documento = carregar_arquivos(tipo_arquivo, arquivo)

    system_message = 'Você e um assistente legal chamado Chat de Julio.' \
    'Você possui acesso as seguintes informações vindas de um documento {}:' \
    '###' \
    '{}' \
    '####' \
    'Utilize as informações fornecidas para basear as suas respostas.' \
    'Sempre que houver um $ na sua saída substitua por S.' \
    'Se a informação do documento for algo como "Just a moment...Enable JavaScript and cookies to continue" sugira ao usuário carregar novamente o chat'.format(tipo_arquivo,documento)
    template = ChatPromptTemplate.from_messages([
        ('system', system_message),
        ('placeholder', '{chat_history}'),
        ('user', '{input}')
    ])

    chat = provedores[provedor]['chat'](model = modelo, api_key=api_key)
    chain = template | chat
    st.session_state['chain'] = chain
    st.session_state['memoria'] = memoria


def home_page():
    st.header('Bem vindo ao ChatGPT de Julio 🤖', divider=True)

    chain = st.session_state.get('chain')
    if chain is None:
        st.error('Para conversar com o Chat de Julio, selecione um tipo de arquivo e inicialize com uma API!')
        st.stop()

    mensagens = st.session_state.get('memoria', memoria)

    for msg in mensagens.buffer_as_messages:
        chat = st.chat_message(msg.type)
        chat.markdown(msg.content)

    input_usuario = st.chat_input('Converse com o Chat de Julio')
    if input_usuario:
        mensagens.chat_memory.add_user_message(input_usuario)
        chat = st.chat_message('human')
        chat.markdown(input_usuario)

        chat = st.chat_message('ai')

        resposta = chain.invoke({'input': input_usuario, 'chat_history': mensagens.buffer_as_messages}).content
        mensagens.chat_memory.add_ai_message(resposta)

        st.session_state['memoria'] = mensagens

def sidebar():
    tabs = st.tabs(['Upload de Arquivo', 'Seleção de Modelos'])
    tipo_arquivo = None
    arquivo = None
    with tabs[0]:
        tipo_arquivo = st.selectbox('Selecione o tipo de arquivo', arquivos_validos)
        if tipo_arquivo == 'Site':
            arquivo = st.text_input('Digite a URL do site:')
        if tipo_arquivo == 'Youtube':
            arquivo = st.text_input('Digite a URL do video:')
        if tipo_arquivo == 'Pdf':
            arquivo = st.file_uploader('Faça o Upload do arquivo PDF', type = ['.pdf'])
        if tipo_arquivo == 'Csv':
            arquivo = st.file_uploader('Faça o Upload do arquivo CSV', type = ['.csv'])
        if tipo_arquivo == 'Txt':
            arquivo = st.file_uploader('Faça o Upload do arquivo TXT', type = ['.txt'])
    with tabs[1]:
        provedor = st.selectbox('Selecione o provedor do chat:', provedores.keys())
        modelo = st.selectbox('Selecione o modelo:', provedores[provedor]['modelos'])
        api_key = st.text_input(f'Adcione a chave de API para o provedor {provedor}', value = st.session_state.get(f'api_key_{provedor}', ))

        st.session_state[f'api_key_{provedor}'] = api_key

    if st.button('Inicializar o Chat de Julio', use_container_width=True):
        if (tipo_arquivo in ['Pdf', 'Csv', 'Txt'] and arquivo is None) or (tipo_arquivo in ['Site', 'Youtube'] and not arquivo):
            st.warning('⚠️ Por favor, informe ou carregue o arquivo antes de iniciar.')
        else:
            carregar_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)
    if st.button('Apagar histórico de conversa', use_container_width=True):
        st.session_state['memoria'] = memoria
        st.session_state['chain'] = None

    st.markdown("""
    <style>
        .rodape {
            position: fixed;
            bottom: 10px;
            left: 0;
            width: 15%;
            text-align: center;
            font-size: 0.9rem;
            color: gray;
        }

        @media (max-width: 1200px) {
            .rodape {
                width: 25%;
            }
        }
    </style>
    <div class='rodape'>
        Desenvolvido por <a href="https://www.linkedin.com/in/jcesarc/" target="_blank">Júlio Cesar</a>
    </div>
""", unsafe_allow_html=True)


def main():
    with st.sidebar:
        sidebar()
    home_page()
    

if __name__ == '__main__':
    main()