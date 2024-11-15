import inspect
import re

class ToolBase():
    @classmethod
    def get_tool_method_name(cls):
        for attribute_name in dir(cls):
            attribute = getattr(cls, attribute_name)
               
            if callable(attribute) and getattr(attribute, 'is_tool_function', False):
                return attribute_name
        return None

    @staticmethod
    def tool_function(func):
        func.is_tool_function = True  # Mark the function with the attribute
        return func


    def generate_function_json_spec(self):
        # Use the method resolution order (MRO) to access the child class's method
        method_name = self.get_tool_method_name()

        method = getattr(self, method_name, None)
        docstring = inspect.getdoc(method)
        parameters = inspect.signature(method).parameters
        
        parameter_descriptions = self._extract_parameters_from_args_section(docstring)

        parameters_json = {
            name: {
                "type": "string",  # You can customize this as needed
                "description": parameter_descriptions.get(name, f"Description for {name}")
            }
            for name in parameters
        }

        function_definition = {
            "type": "function",
            "function": {
                "name": method_name,
                "description": docstring.split("\n")[0],  # First line as a brief description
                "parameters": {
                    "type": "object",
                    "properties": parameters_json,
                    "required": list(parameters_json.keys())
                }
            }
        }

        return function_definition

    def _extract_parameters_from_args_section(self, docstring):
        # Extract the "Args" section from the docstring
        args_section_match = re.search(r"Args:\s*(.*)", docstring, re.DOTALL)
        if not args_section_match:
            return {}

        args_section = args_section_match.group(1)

        # Use regex to extract parameter names and their descriptions within the "Args" section
        matches = re.findall(r"(\w+):\s*(.*?)(?=\n\s*\w+:|\Z)", args_section, re.DOTALL)

        # Organize the results in a dictionary
        parameter_descriptions = {param: desc.strip() for param, desc in matches}
        return parameter_descriptions
    
    def execute_tool(self, *args, **kwargs):
        method_name = self.get_tool_method_name()
        method = getattr(self, method_name, None)
        return method(*args, **kwargs)