import os  
import streamlit as st
import re
from dotenv import load_dotenv  
from openai import AzureOpenAI

import helpers as helpers
from luxury_agents import OrchestratorAgent, GetPradaFidelityDataAgent, RagAgent

# CONSTANTS
BOT_ICON = 'https://865aadc87c3454520411-3632423099c3393f1a8bc0dce61fb95f.ssl.cf1.rackcdn.com/contoso-logo-transparent.png'
APP_TITLE = 'Demo: Multi-Agent Orchestration'
CHAT_INPUT_PLACEHOLDER = "Your message: "
CUSTOMER_LOGO = 'https://865aadc87c3454520411-3632423099c3393f1a8bc0dce61fb95f.ssl.cf1.rackcdn.com/contoso-logo-transparent.png'
SPINNER_MESSAGE = "Preparando la tua risposta..."
USER_ICON = 'https://static.vecteezy.com/system/resources/previews/014/194/215/non_2x/avatar-icon-human-a-person-s-badge-social-media-profile-symbol-the-symbol-of-a-person-vector.jpg'

CUSTOMER_LOGO="https://logo.com/image-cdn/images/kts928pd/production/5be7f05ad50b4254e440898461e4ad1026a11723-900x592.png?w=1080&q=80"
BOT_ICON = CUSTOMER_LOGO

logger = helpers.configure_logger("orchestrator_script")

# Load environment variables from the .env file  
load_dotenv(override=True)
 
az_openai_client = AzureOpenAI(  
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)  

llm_config = {
    'llm_type': 'aoai',
    'llm_client': az_openai_client,
    'llm_deployment_name': os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
}

# Declare the agents 
############################# CHANGE HERE FOR CUSTOMIZATION
worker_agents_list = [
    GetPradaFidelityDataAgent("get_prada_fidelity_data_agent", llm_config=llm_config),
    #OrderAgent("customer_order_agent", llm_config=llm_config),
    RagAgent("generic_information_agent", llm_config=llm_config)#,
    #BillingAgent("billing_agent", llm_config=llm_config)
]

orchestrator_agent = OrchestratorAgent("orchestrator_agent", worker_agents_list, llm_config=llm_config)
########################################### WHAT GOES BELOW SHALL NOT BE CHANGED


worker_agents = {agent.get_agent_name(): agent for agent in worker_agents_list}


# Función para mostrar mensajes en forma de bocadillo
def get_message_markdown(message, message_role="user"):
    if message_role == "user":
        #st.markdown(f'<div style="background-color: lightblue; border-radius: 10px; padding: 10px; color: black; text-align: right; margin-left: 50%;">{message}</div>', unsafe_allow_html=True)
        return f"""
            <div style="display: flex; align-items: center; justify-content: flex-end;">
                    <div style="background-color: lightblue; border-radius: 10px; padding: 10px; color: black; text-align: right; margin-left: 50%;">
                        {message}
                    </div>
                    <img src="{USER_ICON}" alt="User Icon" style="width: 40px; height: 40px; margin-left: 10px;">
            </div>
            """
    else:
        return f"""
            <div style="display: flex; align-items: center;">
                <img src="{BOT_ICON}" alt="Contoso Logo" style="width: 70px; margin-right: 10px;">
                <div style="background-color: #FF4E00; border-radius: 10px; padding: 10px; color: white; text-align: left; margin-right: 50%;">
                    {message}
                </div>
            </div>
            """


def store_message(message, is_user=True):
    message_role = "user" if is_user else "assistant"
    st.session_state.messages.append({"role": message_role, "content": message})


def update_agent(response_message):
    next_agent_name = response_message.split("=")[1].strip()
    next_agent_name = re.sub(r'[^a-zA-Z0-9_-]', '', next_agent_name)

    if next_agent_name == st.session_state.orchestrator_agent.get_agent_name():
        st.session_state.current_agent = st.session_state.orchestrator_agent 
    else: 
        st.session_state.current_agent = worker_agents[next_agent_name]
    log_agent_orchestration(f"Updated agent: {st.session_state.current_agent.get_agent_name()}")  


def run_conversation():  
    # Step 1: Orchestrate the initial call
    response_from_agent = st.session_state.current_agent.invoke_agent(st.session_state.messages)

    logger.info(f"Response message: {response_from_agent}")

    # The orchestrator agent will orchestrate the conversation based on the user's message
    while 'next_agent=' in response_from_agent:
        update_agent(response_from_agent)
        response_from_agent = st.session_state.current_agent.invoke_agent(st.session_state.messages)

    return response_from_agent

def log_agent_orchestration(message):
    logger.info(message)
    st.session_state.agent_orchestration_logs.append(message)

# MAIN
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.agent_orchestration_logs = []

    st.session_state.orchestrator_agent = orchestrator_agent
    st.session_state.worker_agents = worker_agents
    st.session_state.current_agent = st.session_state.orchestrator_agent
    log_agent_orchestration(f"Initial agent: {st.session_state.current_agent.get_agent_name()}")

st.set_page_config(
    page_title=APP_TITLE,
    layout="centered",
    initial_sidebar_state="auto",
)

st.image(CUSTOMER_LOGO, width=80)
st.title(APP_TITLE)

user_input = st.chat_input(CHAT_INPUT_PLACEHOLDER, key="input")

if user_input:
    store_message(user_input)

    with st.spinner(SPINNER_MESSAGE):
        assistant_response = run_conversation()
        store_message(assistant_response, is_user=False)

    # Mostrar todos los mensajes almacenados en la sesión
    for message in st.session_state.messages:
        message_role = message["role"]
        message_content = message["content"]

        if message_role in ("user", "assistant", "function") and message_content:
            logger.debug(f"Printing message: {message_content} of role: {message_role}")
            message_markdown = get_message_markdown(message_content, message_role)
            st.markdown(message_markdown, unsafe_allow_html=True)

st.sidebar.markdown(f"<h1 style='font-weight: bold;'>Agent Orchestration logs</h1>", unsafe_allow_html=True)
st.sidebar.code("\n".join(st.session_state.agent_orchestration_logs))
