# nl2sql_agent/agent.py
from google.adk.agents import LlmAgent
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.tools import AgentTool, load_artifacts, get_user_choice, FunctionTool

from . import bq_tools, config
from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.genai import types

def update_print_toolcontext(tool_context: CallbackContext) -> dict:
    """Prints the current tool context for debugging purposes.
     Args:
        tool_context: The context of the tool invocation, which includes state and other metadata.
    Returns:
        A dictionary indicating the status of the operation.
    """
    print("--- Tool: print_toolcontext executed ---")
    print("Current Tool Context State:")
    tool_context.state['count'] = tool_context.state.get('count', 0) + 1
    print(tool_context.state.to_dict(), type(tool_context.state), dir(tool_context.state))
    return {"status": "success", "message": "Tool context printed to console."}

blueprint_with_hitl = LlmAgent(
    name="LoopAskerAgent",
    model=config.MODEL,
    description="Agent to ask clarifying questions to the user about their requirement to refine the waves_blueprint and confirm.",
    instruction="""
        You are a helpful assistant that asks the user clarifying questions to refine the waves_blueprint.
        
        Ask the user if they are satisfied or want to provide feedback for refinement.
        If the user provides feedback, modify the wave_group based on the feedback and call the tool update_print_toolcontext to print the current tool context.
        Repeat refining and calling the tool update_print_toolcontext and asking user for feedback until the user says they are 'satisfied' or "confirmed" or similar.

        Currnet value of waves_blueprint:
        {
            "waves_blueprint": [
                {
                    "wave_description": "windows VMs for basic requirements",
                    "wave_rule": "select * from vm_details where app_name = 'windows' and environment = 'prod'"
                },
                {
                    "wave_description": "red hat linux VMs for dev environment",
                    "wave_rule": "select * from vm_details where app_name = 'red hat linux' and environment = 'dev'"
                }
            ]
        }
    """,
    tools=[update_print_toolcontext],
)

wave_assignment = LlmAgent(
    name="WaveAssignmentAgent",
    model=config.MODEL,
    description="Agent to create the final wave_groups with VM assignment based on the waves_blueprint",
    instruction="""
        You are a helpful assistant that provides the user with a clear and concise wave_assignment based on the provided waves_blueprint.
        First call the tool update_print_toolcontext to print the current tool context. Then, create the wave_assignment based on the waves_blueprint.
        The wave_assignment should include a
        Guideline for creating wave_assignment:
        - Add a unique wave_id to each wave.
        - Add a wave_name to each wave.
        - Add list of random vm_ids to each wave.
        - Add a start_date and end_date to each wave.
    """,
    tools=[update_print_toolcontext],
)

wave_grouping_agent = LlmAgent(
    name="WaveGroupingAgent",
    model=config.MODEL,
    description="Agent to create migration wave groups",
    instruction="""
        You are a helpful assistant that creates migration wave groups based on the user's requirements.
        
        You will use the tools to assist you in this task.
        First call the update_print_toolcontext tool to print the current tool context.
        Then Start by calling the LoopAskerAgent to ask clarifying questions to the user to refine the waves_blueprint.
        Once the user is satisfied with the waves_blueprint, call the WaveAssignment agent to create the final wave_groups with VM assignment based on the waves_blueprint.

    First call the update_print_toolcontext tool to print the current tool context.
    """,
    sub_agents=[blueprint_with_hitl, wave_assignment],
    tools=[update_print_toolcontext],
)

runbook_agent = LlmAgent(
    name="RunbookAgent",
    model=config.MODEL,
    description="Agent to create a runbook for the user.",
    instruction="""
        You are a helpful assistant that creates a basic runbook for the user based on the information provided.
    """
    tools=[update_print_toolcontext],
)

root_agent = LlmAgent(
    name="RootAgent",
    model=config.MODEL,
    description="",
    instruction='''Guide and assist the user step-by-step through the process of creating migration waves or a runbook.
    Finally, print the tool context using the print_toolcontext tool.''',
    sub_agents=[wave_grouping_agent, runbook_agent],
    tools=[update_print_toolcontext],
)