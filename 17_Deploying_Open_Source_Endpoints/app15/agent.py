import os

from collections.abc import AsyncIterable
from typing import Any, Literal

from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from app.agent_graph_with_helpfulness import build_agent_graph_with_helpfulness


memory = MemorySaver()

class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


class Agent:
    """Agent - a general-purpose assistant with access to web search, academic papers, and RAG."""

    SYSTEM_INSTRUCTION = (
        'You are an expert NPTE (National Physical Therapy Examination) question writer. '
        'You have access to various tools including web search, academic paper search, and document retrieval. '
        'IMPORTANT: Always use the available tools (Tavily, ArXiv, RAG, Google Scholar, IJSPT) to research current information '
        'before creating NPTE-style questions. This ensures your questions are based on current best practices and evidence. '
        'Use the appropriate tools to research current information and create high-quality NPTE-style questions. '
        'Always format your responses as NPTE-style multiple choice questions with A, B, C, D options.'
    )

    FORMAT_INSTRUCTION = (
        'Your responses should be in the form of either a direct question with four multiple choice answers (A, B, C, D), '
        'or provide a realistic clinical scenario with a question and 4 multiple choice answers. '
        'Ensure the question follows 2023-2025 NPTE standards and uses current best practices as of 2025. '
        'Make scenarios realistic and clinically relevant with exactly 4 plausible answer choices where only one is correct.'
    )

    def __init__(self):
        self.model = ChatOpenAI(
            model=os.getenv('TOOL_LLM_NAME', 'gpt-4o-mini'),
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            openai_api_base=os.getenv('TOOL_LLM_URL', 'https://api.openai.com/v1'),
            temperature=0,
        )
        # Use the new graph with helpfulness evaluation for A2A protocol compatibility
        self.graph = build_agent_graph_with_helpfulness(
            self.model,
            self.SYSTEM_INSTRUCTION,
            self.FORMAT_INSTRUCTION,
            checkpointer=memory
        )

    async def stream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        inputs = {'messages': [('user', query)], 'loop_count': 0}
        config = {'configurable': {'thread_id': context_id}}

        pre_helpfulness_response = None
        final_response = None

        for item in self.graph.stream(inputs, config, stream_mode='values'):
            message = item['messages'][-1]
            
            # Capture the first AI response (pre-helpfulness)
            if (isinstance(message, AIMessage) and 
                not message.content.startswith("HELPFULNESS:") and
                not getattr(message, "tool_calls", None) and
                pre_helpfulness_response is None):
                pre_helpfulness_response = message.content
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': f"Pre-helpfulness response:\n{pre_helpfulness_response}\n\nEvaluating helpfulness...",
                }
            
            elif (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Searching for information...',
                }
            elif isinstance(message, ToolMessage):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Processing the results...',
                }
            elif (isinstance(message, AIMessage) and 
                  message.content.startswith("HELPFULNESS:")):
                # This is a helpfulness evaluation
                helpfulness_result = message.content
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': f"Helpfulness evaluation: {helpfulness_result}",
                }

        # Get the final state to extract the complete response
        final_state = self.graph.get_state(config)
        messages = final_state.values.get('messages', [])
        
        # Find the final non-helpfulness AI message
        for msg in reversed(messages):
            if (isinstance(msg, AIMessage) and 
                not msg.content.startswith("HELPFULNESS:") and
                not getattr(msg, "tool_calls", None)):
                final_response = msg.content
                break

        # Return the comparison
        if pre_helpfulness_response and final_response:
            comparison_content = f"""
=== PRE-HELPFULNESS RESPONSE ===
{pre_helpfulness_response}

=== FINAL RESPONSE (AFTER HELPFULNESS EVALUATION) ===
{final_response}

=== COMPARISON ===
The helpfulness evaluation loop has refined the response to better meet NPTE standards.
"""
        else:
            comparison_content = final_response or "No response generated."

        yield {
            'is_task_complete': True,
            'require_user_input': False,
            'content': comparison_content,
        }

    def get_agent_response(self, config):
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get('structured_response')
        if structured_response and isinstance(
            structured_response, ResponseFormat
        ):
            if structured_response.status == 'input_required':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'error':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'completed':
                return {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': structured_response.message,
                }

        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': (
                'We are unable to process your request at the moment. '
                'Please try again.'
            ),
        }

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']


def graph():
    """Zero-argument graph builder that exposes the main agent graph."""
    model = ChatOpenAI(
        model=os.getenv('TOOL_LLM_NAME', 'gpt-4o-mini'),
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('TOOL_LLM_URL', 'https://api.openai.com/v1'),
        temperature=0,
    )
    return build_agent_graph_with_helpfulness(
        model,
        Agent.SYSTEM_INSTRUCTION,
        Agent.FORMAT_INSTRUCTION,
        checkpointer=memory,
    )
