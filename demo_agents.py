import json
import os
import pytz  
import random
import requests
from datetime import datetime  
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import AzureOpenAI 

import azure.cosmos.cosmos_client as cosmos_client
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

from iagents import LLMBaseAgent
from iagents.agent_tools import ToolBase


class ToolGetSmartletData(ToolBase):
    @ToolBase.tool_function
    def get_Smartlet_data(self, email, type_of_data):
        """
        Get the Smartlet customer data (balance, cupons or benefits) for a given email.

        Args:
            email: The user's email. User's email should be in a format like name@domain.com.
            type_of_data: balance, cupons or benefits associated to the Smartlet card.
        """
        
        try:
            # Cargar los datos
            filename = 'data/Smartlet_data.json'
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
                    return user_data["saldo"]
                elif type_of_data == 'cupons':
                    cupones=f'Tienes {len(user_data["cupones"])} cupones:'
                    for cupon in user_data["cupones"]:
                        cupones+=f'\n- Tipo: {cupon["tipo"]}, Descripción: {cupon["descripcion"]}, Fecha de expiración: {cupon["fecha_expiracion"]}\n'
                    return cupones
                elif type_of_data == 'benefits':
                    beneficios=f'Tienes {len(user_data["beneficios"])} beneficios:'
                    for beneficio in user_data["beneficios"]:
                        beneficios+=f'\n- Tipo: {beneficio["comercio"]}, Descripción: {beneficio["descripcion"]}, Beneficio: {beneficio["beneficio"]}\n'
                    return beneficios
            else:
                return f"Sorry, I couldn't find the data for the user's email {email}."
        
        except:
            return f"Sorry, I couldn't find the data for the user's email {email}."


class ToolOrder(ToolBase):
    @ToolBase.tool_function
    def place_order(self, task, due, email):
        """
        Write an email with the work order. Include information collected from the chat.

        Args:
            task: Use information from the chat for the work order
            due: The postal address in a format like Avda. de Madrid, 61 2 2 28100 Alcobendas (Madrid).
            email: The number of gas cylinder to order.
        """
        
        #try:
        order_num = random.randint(100, 10000)
        date_plus_one = datetime.now() + timedelta(1)
        date = date_plus_one.strftime('%d-%m-%Y')

        # Cargar configuración de CosmosDB
        cosmos_host=os.getenv("COSMOS_HOST")
        cosmos_key=os.getenv("COSMOS_KEY")
        cosmos_database=os.getenv("COSMOS_DATABASE")
        cosmos_container=os.getenv("COSMOS_CONTAINER")

        # Creo la instancia del cliente, DB y Container
        client = cosmos_client.CosmosClient(cosmos_host, {'masterKey': cosmos_key}, user_agent="CosmosDBPythonQuickstart", user_agent_overwrite=True)
        db = client.get_database_client(cosmos_database)
        container = db.get_container_client(cosmos_container)
        sales_order = {
            "id": str(order_num),
            "task": task,
            "due": due,
            "email": email,
            "date": date
        }
        # Cargar los datos en CosmosDB
        container.create_item(body=sales_order)

        response = f"Order information: Order number: {order_num}, Product: {product}, Address: {address}, Date:{date}"
        #self.order_num += 1
        print(f'\nRESPONSE: {response}\n')

        # Enviar el email con Logic App
        logic_app_url=os.getenv("LOGIC_APP_URL")
        http_response = requests.post(logic_app_url, json=sales_order)
        http_response.raise_for_status()

        return response


class ToolGetTime(ToolBase):
    @ToolBase.tool_function
    def get_current_time(self, location):
        """
        Get the current time in a given location.

        Args:
            location: The location name. The pytz is used to get the timezone for that location. Location names should be in a format like America/New_York, Asia/Bangkok, Europe/London
        """
        
        try:
            timezone = pytz.timezone(location)  
            now = datetime.now(timezone)  
            current_time = now.strftime("%I:%M:%S %p")  
            return current_time  
        except:  
            return "Sorry, I couldn't find the timezone for that location."  


class ToolRag(ToolBase):
    @ToolBase.tool_function
    def search_index(self, query):
        """
        Get general information about Contoso from an Azure cognitive search index.

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


class ToolSergioTime(ToolBase):
    @ToolBase.tool_function
    def get_time_in_location(self, location, summer_time_saving):
        """
        Get the current time in a given location.

        Args:
            location: The location of interest
            summar_time_saving: If the location does summer time changes
        """
        return f"Time in {location} can be tracked with a watch. It is {summer_time_saving} that they change time"


class ToolSergioStock(ToolBase):
    @ToolBase.tool_function
    def get_stock_price(self, stock_name):
        """
        Get the price of a stock.

        Args:
            stock_name: The name of the stock
        """
        return f"The stock in {stock_name} can be paid with money"


class BillingAgent(LLMBaseAgent):

    agent_description = "Expert in customer billing and invoicing data. You don't provide generic service information but only that specific to a given customer billing."

    base_prompt = f"""
    You are an agent {agent_description}.
    Mock up customer data for billing information in euros
    """


class OrderAgent(LLMBaseAgent):

    agent_description = "Expert sending Oil Work Orders that were generated"

    base_prompt = f"""
    You are an agent {agent_description}.
    In order to inform a team about a work order on oil and gas, you can use the available tools. To create the work order use the information that you have received in the chat upfront.
    """
    
   # tools = [ToolOrder()]


class TimeAgent(LLMBaseAgent):

    agent_description = "Expert in time retrieval."

    base_prompt = f"""
    You are an agent {agent_description}. 
    In order to get the current time in a location, you can use the available tools. You should collect from the user the location for which they want to know the current time.
    """
    
    tools = [ToolGetTime()]


class OrchestratorAgent(LLMBaseAgent):
    base_prompt = """
    You are an agent orchestrator for Contoso customers. Contoso is an energy company, selling fuel, electricity, gas.
    Contoso products and brands are Smartlet, Contored, and Contoso. You are responsible for orchestrating the conversation between the user and the worker agents.
    If you need to orchestrate to a worker agent, based on the customer input, indicate which agent you would like to orchestrate to.
    If the user does not provide enough information, ask the user to rephrase the question or interact with the user until it gives a clear answer.
    You will be given a list of agents to choose from. Each agent entry in the list will consist of a name and a description.
    As an example:
    - weather_agent: This agent provides weather
    - stock_agent: This agent provides stock information
    In order to indicate the output agent. Generate output like "next_agent=agent_name"
    For example, if the user asks about the weather, you can generate "next_agent=weather_agent"
    Only use the agent names provided in the list below. If no agent is available, say you are sorry and ask the user to rephrase the question.
    You will orchestrate the conversation based on the user's message and only the following agents and agents description: \n
    """
    
    def __init__(self, agent_name: str, worker_agent_list: list, **kwargs):
        prompt_extension = ""
        for worker_agent in  worker_agent_list:
            prompt_extension += f"- Agent Name == {worker_agent.get_agent_name()}, Agent Purpose Description == {worker_agent.get_agent_description()}\n"

        super().__init__(agent_name, prompt_extension, is_orchestrator=True, **kwargs)


class GetSmartletDataAgent(LLMBaseAgent):

    agent_description = "Expert in Smartlet customer data retrieval. You don't provide generic Smartlet information."

    base_prompt = f"""
    You are an agent {agent_description}.
    In order to get the balance, cupons and benefits of an user of a Smartlet card, you can use the available tools. You should collect from the user the email and the type of data to retrieve: balance, cupons or benefits.
    """
    
    tools = [ToolGetSmartletData()]


class RagAgent(LLMBaseAgent):

    agent_description = "Expert in providing generic information about Contoso."

    base_prompt = f"""
    You are an agent {agent_description}.You have access to an Azure Cognitive Search index with this information. 
    You are designed to be an interactive assistant, so you can ask users clarifying questions to help them find the information. It's better to give more detailed queries to the search index rather than vague one.
    """
    
    tools = [ToolRag()]