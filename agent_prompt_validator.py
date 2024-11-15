
import os  
import pandas as pd  
import re
from dotenv import load_dotenv  
from openai import AzureOpenAI

import helpers as helpers
from colorama import Fore, Style
from orchestrator_script import invoke_agent
from demo_agents import OrchestratorAgent, TimeAgent, GetWayletDataAgent, OrderAgent, RagAgent, BillingAgent

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, message="missing ScriptRunContext")

ground_truth_file = 'agent_swap_ground_truth.csv'

# Declare the agents and functions
time_agent = TimeAgent("time_agent")
get_waylet_data_agent = GetWayletDataAgent("get_waylet_data_agent")
order_agent = OrderAgent("customer_order_agent")
rag_agent = RagAgent("generic_information_agent")
billing_agent = BillingAgent("billing_agent")

worker_agent_list = [time_agent, get_waylet_data_agent, order_agent, rag_agent, billing_agent]
orchestrator_agent = OrchestratorAgent("orchestrator_agent", worker_agent_list)

logger = helpers.configure_logger("orchestrator_script")

# Load environment variables from the .env file  
load_dotenv(override=True)
 
model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
client = AzureOpenAI(  
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)  


ground_truth_data = pd.read_csv(ground_truth_file)

for _, row in ground_truth_data.iterrows():
    question = row['question']
    expected_agent = row['agent']

    question_message = [{'content': question, 'role': 'user'}]
    agent_response = invoke_agent(orchestrator_agent, question_message)

    response_ok = False
    if 'next_agent' in agent_response:
        next_agent_name = agent_response.split("=")[1].strip()
        next_agent_name = re.sub(r'[^a-zA-Z0-9_-]', '', next_agent_name)

        if next_agent_name == expected_agent:
            response_ok = True

    if response_ok:
        print(f"{Fore.GREEN}OK{Style.RESET_ALL} - Question: {question}, Expected Agent: {next_agent_name}")
    else:
        print(f"{Fore.RED}KO{Style.RESET_ALL} - Question: {question}, Expected Agent: {expected_agent}, Answer: {agent_response}")