import json
import os
from dotenv import load_dotenv
from openai import AzureOpenAI 

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
import requests

from iagents import LLMBaseAgent
from iagents.agent_tools import ToolBase


class ToolGetPradaFidelityData(ToolBase):
    @ToolBase.tool_function
    def get_PradaFidelity_data(self, email, type_of_data):
        """
        Get the PradaFidelity customer data (balance, cupons or benefits) for a given email.

        Args:
            email: The user's email. User's email should be in a format like name@domain.com.
            type_of_data: balance, cupons or benefits associated to the PradaFidelity card.
        """
        
        try:
            # Cargar los datos
            filename = 'data/PradaFidelity_data.json'
            with open(filename, 'r', encoding="utf-8") as f:
                users = json.load(f)
            
            # Buscar el usuario por su email
            user_data = None
            for user in users:
                if user["email"] == email:
                    user_data = user

            # Preparo los datos de respuesta
            if user_data != None:
                if type_of_data == 'balance':
                    return user_data["balance"]
                elif type_of_data == 'coupons':
                    coupons=f'You have {len(user_data["coupons"])} coupons:'
                    for coupon in user_data["coupones"]:
                        coupons+=f'\n- Type: {coupon["tipo"]}, Description: {coupon["description"]}, Expiration date: {coupon["expiration_date"]}\n'
                    return coupons
                elif type_of_data == 'benefits':
                    benefits = f'you have {len(user_data["benefits"])} benefits:'
                    for benefit in user_data["benefits"]:
                        benefits+=f'\n- Type: {benefit["store"]}, description: {benefit["description"]}, Benefit: {benefit["benefit"]}\n'
                    return benefits
            else:
                return f"Sorry, I couldn't find the data for the user's email {email}."
        
        except:
            return f"Sorry, I couldn't find the data for the user's email {email}."


class ToolRag(ToolBase):
    @ToolBase.tool_function
    def search_index(self, query):
        """
        Get general information about Prada from an Azure cognitive search index.

        Args:    
            query: The user query string.
        """
        
        # try:

        load_dotenv(override=True)

        search_client = SearchClient(
            endpoint=os.getenv("AI_SEARCH_SERVICE_ENDPOINT"), 
            index_name=os.getenv("INDEX_NAME"), 
            credential=AzureKeyCredential(os.getenv("AI_SEARCH_SERVICE_QUERY_KEY"))
            )   

        aoai_client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_EMBEDDINGS"),  # The base URL for your Azure OpenAI resource. e.g. "https://<your resource name>.openai.azure.com"
            api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_KEY"),  # The API key for your Azure OpenAI resource.
            api_version=os.getenv("OPENAI_API_VERSION")  # This version supports function calling
            )

        query_embeddings = aoai_client.embeddings.create(input=query, model=os.getenv("AZURE_OPENAI_EMBEDDING_NAME_ADA"))
        embeddings = query_embeddings.data[0].embedding

        results = search_client.search(
            query_type="semantic",
            semantic_configuration_name="my-semantic-config",
            search_text=query,
            vector_queries=[VectorizedQuery(vector=embeddings, k_nearest_neighbors=1, fields="question_embeddings")],
            select=["answer"],
            top=1,
        )
        
        answer_for_prompt = ""
        
        for result in results:
            answer_for_prompt += ' '+ result['answer']

        return answer_for_prompt


class ToolTriggerLogicApp(ToolBase):
    @ToolBase.tool_function
    def trigger_logic_app(self, email_content):
        """
        Send the last text in order to send a notification email.

        Args:
            email_content: The content of the email to be sent in markdown format. 
        """
        
        logic_app_url = os.getenv("LOGIC_APP_URL")
        #logic_app_url = "https://prod-17.swedencentral.logic.azure.com:443/workflows/8514ea516d73452fb97a78f93ea25e4d/triggers/When_a_HTTP_request_is_received/paths/invoke?api-version=2016-10-01&sp=%2Ftriggers%2FWhen_a_HTTP_request_is_received%2Frun&sv=1.0&sig=VDNrVK5QtjQgxcrMA-mogIxQ769iAIAgl8BB_qnTqlY"
        if not logic_app_url:
            raise ValueError("LOGIC_APP_URL is not defined in the environment variables.")
        
        response = requests.post(logic_app_url, json=email_content)
        if response.status_code in ["200", "202"]:
            raise Exception(f"Failed to trigger Logic App: {response.status_code} {response.text}")
        else:
            return "Email has been successfully sent"
 

class OrchestratorAgent(LLMBaseAgent):
    base_prompt = """
    You are an agent orchestrator that orchestrates user interactions towards other ai agents.
    You are responsible for orchestrating the conversation between the user and the worker agents.
    ################ CUSTOMER SCENARIO ################
    You work for Siemens, a global leader in industrial manufacturing and supply chain solutions. 
    Your primary focus is optimizing logistics, improving supplier performance, and enhancing production efficiency.
    ################ ORCHESTRATOR LOGIC ################
    If you need to orchestrate to a worker agent, based on the customer input, indicate which agent you would like to orchestrate to.
    If the user does not provide enough information, ask the user to rephrase the question or interact with the user until it gives a clear answer.
    You will be given a list of agents to choose from. Each agent entry in the list will consist of a name and a description.
    As an example:
    - demand_forecast_agent: Forecasts demand and recommends production adjustments.
    - supply_chain_risk_agent: Analyzes supply chain risks.
    - production_optimization_agent: Suggests efficiency improvements.
    - energy_efficiency_agent: Optimizes energy consumption.
    - compliance_agent: Ensures regulatory adherence.    
    In order to indicate the output agent. Generate output like "next_agent=agent_name"
    For example, if the user asks about the weather, you can generate "next_agent=weather_agent"
    Only use the agent names provided in the list below. If no agent is available, say you are sorry and ask the user to rephrase the question.
    You will orchestrate the conversation based on the user's message and only the following agents and agents description: \n
    """
   
    def __init__(self, agent_name: str, worker_agent_list: list, **kwargs):
        prompt_extension = ""
        for worker_agent in worker_agent_list:
            prompt_extension += f"- Agent Name == {worker_agent.get_agent_name()}, Agent Purpose Description == {worker_agent.get_agent_description()}\n"

        super().__init__(agent_name, prompt_extension, is_orchestrator=True, **kwargs)


class GetDemandForecastAgent(LLMBaseAgent):
    agent_description = "Expert in demand forecasting and supply chain planning for Siemens’ manufacturing operations."
    
    base_prompt = f"""
    You are an agent {agent_description}.
    You analyze customer demand patterns, predict inventory needs, and suggest production adjustments.
    Feel free to make data up and respond in a table if appropriate.
    Example insights:
    - Product X demand is expected to increase by 15% next quarter due to new EU regulations.
    - Recommend increasing production output by 10% to meet forecasted demand.
    """
    tools = [ToolTriggerLogicApp()]

class SupplyChainRiskAgent(LLMBaseAgent):
    agent_description = "Expert in supply chain risk analysis and mitigation."
    
    base_prompt = f"""
    You are an agent {agent_description}.
    You assess risks such as geopolitical instability, weather disruptions, and supplier reliability.
    Feel free to make data up and respond in a table if appropriate.
    Example insights:
    - Factory closures in China may impact semiconductor supply.
    - Recommend diversifying suppliers for Component Y to mitigate risk.
    """
    tools = [ToolTriggerLogicApp()]

class ProductionOptimizationAgent(LLMBaseAgent):
    agent_description = "Expert in optimizing production processes to maximize efficiency and reduce costs."
    
    base_prompt = f"""
    You are an agent {agent_description}.
    You analyze production data and suggest improvements.
    Feel free to make data up and respond in a table if appropriate.
    Example insights:
    - Reducing machine idle time by 8% can increase overall efficiency.
    - Implementing predictive maintenance on critical machines can reduce downtime.
    You do not answer questions about sustainability and suppliers.
    """
    tools = [ToolTriggerLogicApp()]

class EnergyEfficiencyAgent(LLMBaseAgent):
    agent_description = "Expert in energy efficiency for manufacturing operations."
    
    base_prompt = f"""
    You are an agent {agent_description}.
    You analyze energy consumption and suggest cost-saving strategies.
    Feel free to make data up and respond in a table if appropriate.
    Example insights:
    - Switching to LED lighting can reduce factory energy costs by 12%.
    - Recommend optimizing HVAC system scheduling to lower power consumption.
    """
    tools = [ToolTriggerLogicApp()]

class ComplianceAgent(LLMBaseAgent):
    agent_description = "Expert in regulatory compliance for Siemens' operations."
    
    base_prompt = f"""
    You are an agent {agent_description}.
    You ensure adherence to industry regulations and standards.
    Feel free to make data up and respond in a table if appropriate.
    Example insights:
    - Product Z meets new EU environmental compliance standards.
    - Recommend updating documentation to align with ISO 9001.
    """
    tools = [ToolTriggerLogicApp()]

class SustainabilityAgent(LLMBaseAgent):
    agent_description = "Expert in sustainability practices for Siemens' operations."
    
    base_prompt = f"""
    You are an agent {agent_description}.
    You provide insights on reducing carbon footprints, sustainable sourcing, and regulatory sustainability compliance.
    Feel free to make data up and respond in a table if appropriate.
    Example insights:
    - Implementing circular economy principles can reduce waste by 20%.
    - Switching to renewable energy sources can lower carbon emissions by 30%.
    You do not answer questions about production cost and suppliers.
    """
    tools = [ToolTriggerLogicApp()]


class RagAgent(LLMBaseAgent):
    agent_description = "You generate reports on supply chain performance, cost efficiency, and demand forecasting for Siemens."

    base_prompt = f"""
    You are an agent {agent_description}.
    You summarize discussions into detailed reports. Feel free to make numbers upa and answer in a table (if suitable).
    You only answer questions in your domain.
    Example report summary:
    - Supplier performance declining by 10% due to shipping delays.
    - Inventory levels optimal, but distribution needs to be balanced across warehouses.
    - Cost-saving potential of 8% identified in procurement strategy.
    """
    tools = [ToolTriggerLogicApp()]


def main():

    # Example usage of the `trigger_logic_app` method
    # Replace 'Your email content here' with the content you want to send
    email_content = "Hello my friend"
    # Initialize the ToolTriggerLogicApp
    trigger_logic_app_tool = ToolTriggerLogicApp()
    # Call the `trigger_logic_app` method
    try:
        result = trigger_logic_app_tool.trigger_logic_app(email_content=email_content)
        print(result)
    except Exception as e:
        print(f"Error occurred: {e}")
# Ensure the script can be executed
if __name__ == "__main__":
    main()

 
   