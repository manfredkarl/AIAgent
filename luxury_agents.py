import json
import os
from dotenv import load_dotenv
from openai import AzureOpenAI 

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

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


class OrchestratorAgent(LLMBaseAgent):
    base_prompt = """
    You are an agent orchestrator that orchestrates user interactions towards other ai agents.
    You are responsible for orchestrating the conversation between the user and the worker agents.
    ################ CUSTOMER SCENARIO ################
    You work for Prada customers. Prada is a fashion company, selling clothing, fashion and complements.
    Prada brands products and brands are PradaFidelity, Contored, and Prada.
    ################ ORCHESTRATOR LOGIC ################
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


class GetPradaFidelityDataAgent(LLMBaseAgent):

    agent_description = "Expert in PradaFidelity customer data retrieval. You don't provide generic PradaFidelity information."

    base_prompt = f"""
    You are an agent {agent_description}.
    In order to get the balance, cupons and benefits of an user of a PradaFidelity card, you can use the available tools. You should collect from the user the email and the type of data to retrieve: balance, cupons or benefits.
    """
    
    tools = [ToolGetPradaFidelityData()]


class RagAgent(LLMBaseAgent):

    agent_description = "Expert in providing generic information about Prada."

    base_prompt = f"""
    You are an agent {agent_description}.You have access to an Azure Cognitive Search index with this information. 
    You are designed to be an interactive assistant, so you can ask users clarifying questions to help them find the information. It's better to give more detailed queries to the search index rather than vague one.
    """
    
    tools = [ToolRag()]