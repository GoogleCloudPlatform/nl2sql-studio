from vertexai import generative_models
from vertexai.generative_models import (
    GenerativeModel,
    Part,
    Tool,
    # ToolConfig
)
import json
import vertexai


agent = GenerativeModel("gemini-1.5-flash-002",
                        generation_config={"temperature": 0.05},
                        )

