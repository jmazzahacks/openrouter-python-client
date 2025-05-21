# Just test direct imports from the models module
try:
    from openrouter_client.models.models import ParameterType, ToolChoice
    print(f"Successfully imported ParameterType: {ParameterType.STRING}")
    print(f"Successfully imported ToolChoice: {type(ToolChoice).__name__}")
    print("Import test successful!")
except Exception as e:
    print(f"Import failed: {e}")