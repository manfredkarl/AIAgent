import json
import logging

logger = logging.getLogger(__name__)

class LLMBaseAgent:
    # Class variables shared across all instances
    agent_description = "This is the base agent class."
    base_prompt = "This is the base prompt for the self."

    back_to_orchestrator_prompt = """
    Respond always in the language of the user input. If the user opens the conversation in English, respond in English
    You are a specialized worker agent orchestrated by another orchestrator self.
    Once you provide a response, ask the user if the user has any other questions.
    If the user is asking a question that is not within your domain. You should respond with a message to the orchestrator. 
    If the user is asking about an activity that you have been indicated not to do pass on to other agents. 
    If you need to send a message to the orchestrator_agent, only generate a message like "next_agent=orchestrator_agent"
    """


    tools = []

    @classmethod
    def _get_base_prompt(cls) -> str:
        """
        Return the base prompt.

        :return: The base prompt string.
        """
        return cls.base_prompt

    @classmethod
    def get_tools(cls) -> list:
        """
        Return the list of tools.

        :return: A list of tools.
        """
        return cls.tools

    @classmethod
    def has_tools(cls) -> bool:
        """
        Check if the agent has tools.

        :return: True if the agent has tools, False otherwise.
        """
        return len(cls.tools) > 0

    @classmethod
    def get_agent_description(cls) -> str:
        """
        Return the agent description.

        :return: The agent description string.
        """
        return cls.agent_description

    @classmethod
    def get_agent_tool_definition(cls) -> list:
        tool_definition = [ tool.generate_function_json_spec()  for tool in cls.tools]

        return tool_definition

    def __init__(self, agent_name: str, prompt_extension: str = "", is_orchestrator=False, llm_config=None):
        
        self.agent_name = agent_name
        self.agent_prompt = self._get_base_prompt()

        self.llm_config = llm_config

        if prompt_extension:
            self.agent_prompt += prompt_extension

        if is_orchestrator is False:
            self.agent_prompt += f"\n\n{self.back_to_orchestrator_prompt}"
        
    def execute_tool_function(self, tool_method_name: str, *args, **kwargs):
        """
        Execute a tool function.

        :param tool_name: The name of the tool.
        :param parameters: The parameters for the tool function.
        :return: The result of the tool function.
        """
        for tool in self.get_tools():
            if tool.get_tool_method_name() == tool_method_name:
                return tool.execute_tool(*args, **kwargs)
        return None

    def get_agent_name(self) -> str:
        """
        Return the agent name.

        :return: The agent name string.
        """
        return self.agent_name
    
    def get_agent_prompt(self) -> str:
        """
        Return the agent prompt.

        :return: The agent prompt string.
        """
        return self.agent_prompt


    def generate_completion(self, messages, tools=None, tool_choice="auto"):
        args = {
            "model": self.llm_config['llm_deployment_name'],
            "messages": messages
        }

        if tools:
            logger.debug("Tools were provided.")
            args["tools"] = tools
            args["tool_choice"] = tool_choice

        response = self.llm_config['llm_client'].chat.completions.create(**args)
        response_message = response.choices[0].message

        if response_message.content:
            logger.debug(f"Response message has content: {response_message.content}")
            return response_message.content
        else:
            logger.debug("Response message has no content")
            return response_message


    def invoke_agent(self, conversation_messages):
        agent_messages = [{"role": "system", "content": self.get_agent_prompt()}]
        agent_messages += conversation_messages

        logger.debug(f"Agent messages: {agent_messages}")

        if self.has_tools():
            response_message = self.generate_completion(  
                messages = agent_messages,
                tools = self.get_agent_tool_definition()
            )
        else:
            response_message = self.generate_completion(messages = agent_messages)

            # Step 2: Check if the response message has tool calls
        if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
            function_name = response_message.tool_calls[0].function.name
            function_args = json.loads(response_message.tool_calls[0].function.arguments)

            logger.info(f"Function call to - {function_name}")
            logger.info(f"Provided args: {function_args}")
            
            # Step 5: Call the function 
            function_output = self.execute_tool_function(function_name, **function_args)
            logger.info(f"Tool output: {function_output}")
            
            agent_messages += [
                {  
                    "role": response_message.role,  
                    "function_call": {  
                        "name": response_message.tool_calls[0].function.name,  
                        "arguments": response_message.tool_calls[0].function.arguments,  
                    },  
                    "content": None,  
                },  
                {  
                    "role": "function",  
                    "name": function_name,  
                    "content": function_output,  
                }  
            ]  

            agent_response_with_function_call = self.generate_completion(messages=agent_messages)
            return agent_response_with_function_call

        return response_message
