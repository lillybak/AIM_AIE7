"""Agent graph with a post-response helpfulness check loop for A2A protocol compatibility.

After the agent responds, a secondary node evaluates helpfulness ('Y'/'N').
If helpful, end; otherwise, continue the loop or terminate after a safe limit.
"""
from __future__ import annotations

from typing import Dict, Any, Annotated, TypedDict, List

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
import re
import urllib.parse


class AgentState(TypedDict):
    """State schema for agent graphs, storing a message list with add_messages."""
    messages: Annotated[List, add_messages]
    structured_response: Any  # ResponseFormat | None
    loop_count: int  # Track loop iterations


def _filter_references(text: str, fallback_query: str | None = None) -> str:
    """Keep only valid reference lines (URLs or local PDF citations) after 'References:'."""
    if "References:" not in text:
        return text

    head, refs = text.split("References:", 1)
    lines = [ln.strip() for ln in refs.strip().splitlines()]
    valid_lines: list[str] = []
    url_pattern = re.compile(r"^(https?://|www\.)", re.IGNORECASE)
    pdf_pattern = re.compile(r"\.pdf(\b|\s).*", re.IGNORECASE)
    for ln in lines:
        if not ln:
            continue
        if url_pattern.search(ln) or pdf_pattern.search(ln):
            valid_lines.append(ln)
    # If nothing valid, add a single reputable fallback link based on the query
    if not valid_lines and fallback_query:
        q = urllib.parse.quote(f"NPTE {fallback_query}")
        valid_lines.append(f"https://scholar.google.com/scholar?q={q}")
    new_refs = "\n".join(valid_lines)
    return f"{head}References:\n{new_refs}" if new_refs else f"{head}References:\n"


def build_model_with_tools(model):
    """Return a model instance bound to the tool belt."""
    from app.tools import get_tool_belt
    return model.bind_tools(get_tool_belt())


def call_model(state: Dict[str, Any], model) -> Dict[str, Any]:
    """Invoke the model with the accumulated messages and append its response."""
    model_with_tools = build_model_with_tools(model)
    messages = state["messages"]
    
    # Add NPTE-specific system instruction for topic-based queries
    if len(messages) == 1 and isinstance(messages[0], HumanMessage):
        user_topic = messages[0].content.strip()
        
        # Define valid NPTE topics and their variations
        valid_topics = {
            "cardiovascular & pulmonary": ["cardiovascular", "pulmonary", "cardiac", "heart", "vascular", "circulatory", "respiratory", "lung", "breathing", "ventilation", "oxygenation"],
            "musculoskeletal & nervous": ["musculoskeletal", "nervous", "muscle", "skeletal", "bone", "joint", "orthopedic", "neurological", "nerve", "spinal", "cervical", "thoracic", "lumbar"],
            "neuromuscular": ["neuromuscular", "muscle", "nerve", "motor", "neurological", "motor control", "muscle strength", "coordination"],
            "integumentary": ["integumentary", "skin", "wound", "burn", "dermatological", "pressure ulcer", "scar", "healing"],
            "metabolic & endocrine": ["metabolic", "endocrine", "diabetes", "thyroid", "hormone", "glucose", "insulin", "metabolism"],
            "gastrointestinal": ["gastrointestinal", "gi", "digestive", "stomach", "intestine", "bowel", "abdominal"],
            "genitourinary": ["genitourinary", "urinary", "bladder", "kidney", "pelvic", "reproductive"],
            "lymphatic": ["lymphatic", "lymph", "edema", "lymphedema", "swelling", "fluid"],
            "system interactions": ["system interactions", "multisystem", "comorbidity", "interdisciplinary", "complex"],
            "equipment, devices, & technologies": ["equipment", "devices", "technologies", "assistive", "prosthetic", "orthotic", "wheelchair", "walker"],
            "therapeutic modalities": ["therapeutic modalities", "modalities", "ultrasound", "electrical stimulation", "heat", "cold", "traction"],
            "safety & protection": ["safety", "protection", "precautions", "infection control", "fall prevention", "ergonomics"],
            "professional responsibilities": ["professional responsibilities", "ethics", "documentation", "communication", "collaboration", "scope of practice"],
            "research & evidence-based practice": ["research", "evidence-based", "evidence", "clinical practice guidelines", "systematic review", "meta-analysis"]
        }
        
        # Check if user topic is valid
        is_valid_topic = False
        matched_topic = None
        
        for full_topic, variations in valid_topics.items():
            if (user_topic.lower() == full_topic.lower() or 
                any(variation.lower() == user_topic.lower() for variation in variations)):
                is_valid_topic = True
                matched_topic = full_topic
                break
        
        if not is_valid_topic:
            # Return error message with available topics
            topic_list = "\n".join([f"- {topic}" for topic in valid_topics.keys()])
            error_message = f"""
I don't recognize "{user_topic}" as a valid NPTE topic. 

Please use one of these available topics:

{topic_list}

You can also use individual terms like "musculoskeletal" or "nervous" instead of "musculoskeletal & nervous".
"""
            return {"messages": [AIMessage(content=error_message)]}
        
        # This is the initial query - add NPTE context
        npte_prompt = f"""
You are an expert NPTE (National Physical Therapy Examination) question writer. 
The user has provided a topic: "{user_topic}" (which maps to the NPTE domain: "{matched_topic}")

Your task is to create a high-quality NPTE-style multiple choice question about this topic.

CRITICAL REQUIREMENTS:
1. Create a realistic clinical scenario or direct question about {matched_topic}
2. Provide exactly 4 multiple choice answers labeled A., B., C., D. (letters only)
3. Ensure the question follows 2023-2025 NPTE standards
4. Make the scenario realistic and clinically relevant
5. Ensure all answer choices are plausible but only one is correct
6. Use current best practices as of 2025

STRICT FORMATTING RULES - FOLLOW EXACTLY:
- FIRST output the question/scenario and the four answer choices (A-D). 
- NO links, lists, or commentary before the MCQ.
- NO general information or resource lists before the MCQ.
- ONLY after the MCQ, add a "References:" section with clickable links that are SPECIFICALLY relevant to answering the MCQ.
- Do NOT reveal which option is correct.
- Do NOT include unrelated or general links.
- References MUST be either actual clickable URLs (starting with http:// or https://) OR citations to local RAG PDFs (e.g., filename.pdf, page N).

REQUIRED FORMAT:
[Clinical Scenario or Direct Question]
A. [Answer choice]
B. [Answer choice]
C. [Answer choice]
D. [Answer choice]

References:
[ONLY include items that directly support the MCQ content, one per line. Each item MUST be either (a) a clickable URL starting with http(s):// or (b) a local PDF citation in the form 'filename.pdf (page N)'. Do not include plain text titles without a link or PDF citation.]

Use available tools to research current information about {matched_topic} if needed.
"""
        # Create a new message with the enhanced prompt
        enhanced_message = HumanMessage(content=npte_prompt)
        messages = [enhanced_message]
    
    response = model_with_tools.invoke(messages)
    # Post-process references when no tool calls
    if isinstance(response, AIMessage) and not getattr(response, "tool_calls", None):
        content = getattr(response, "content", "")
        if "References:" in content:
            # Try to derive a fallback query from the initial user message
            try:
                fallback_query = state["messages"][0].content if state.get("messages") else None
            except Exception:
                fallback_query = None
            filtered = _filter_references(content, fallback_query)
            response = AIMessage(content=filtered)
        else:
            try:
                topic = state["messages"][0].content if state.get("messages") else ""
            except Exception:
                topic = ""
            fallback = f"https://scholar.google.com/scholar?q={urllib.parse.quote('NPTE ' + topic)}"
            response = AIMessage(content=f"{content.rstrip()}\n\nReferences:\n{fallback}")
    return {"messages": [response]}


def route_to_action_or_helpfulness(state: Dict[str, Any]):
    """Decide whether to execute tools or run the helpfulness evaluator."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "action"
    return "helpfulness"


def helpfulness_node(state: Dict[str, Any], model) -> Dict[str, Any]:
    """Evaluate helpfulness of the latest response relative to the initial query."""
    # Increment loop count
    loop_count = state.get("loop_count", 0) + 1
    
    # If we've exceeded loop limit, short-circuit with END decision marker
    if loop_count > 3:
        return {
            "messages": [AIMessage(content="HELPFULNESS:END")],
            "loop_count": loop_count
        }    

    initial_query = state["messages"][0]
    final_response = state["messages"][-1]
    response_text = final_response.content

    # Pattern-based helpfulness evaluation (no LLM needed)
    def evaluate_helpfulness(text: str):
        """Evaluate if response follows NPTE MCQ format using pattern matching and return decision plus flags."""
        # Check for MCQ format: A., B., C., D. choices
        has_mcq_format = (
            "A." in text and "B." in text and "C." in text and "D." in text
        )
        
        # Check that MCQ appears before any references section
        mcq_before_refs = True
        mcq_index = -1
        if has_mcq_format:
            indices = [i for i in [text.find("A."), text.find("B."), text.find("C."), text.find("D.")] if i != -1]
            mcq_index = min(indices) if indices else -1
        if "References:" in text:
            refs_index = text.find("References:")
            if mcq_index != -1 and refs_index != -1:
                mcq_before_refs = mcq_index < refs_index
        
        # Check for reasonable length (not just a list of links)
        has_substance = len(text.strip()) > 100
        
        # Only inspect content BEFORE the first choice for link/resource spam
        pre_mcq_text = text[:mcq_index] if mcq_index != -1 else text[:200]
        starts_with_resources = any(
            phrase in pre_mcq_text.lower()
            for phrase in ["here are", "resources", "http", "www.", "https://", "search results", "found", "available"]
        )
        
        # Topic relevance setup
        user_topic = initial_query.content.lower().strip()
        topic_keywords = {
            "cardiovascular & pulmonary": ["cardiovascular", "pulmonary", "cardiac", "heart", "vascular", "circulatory", "respiratory", "lung", "breathing", "ventilation", "oxygenation"],
            "musculoskeletal & nervous": ["musculoskeletal", "nervous", "muscle", "skeletal", "bone", "joint", "orthopedic", "neurological", "nerve", "spinal", "cervical", "thoracic", "lumbar"],
            "neuromuscular": ["neuromuscular", "muscle", "nerve", "motor", "neurological", "motor control", "muscle strength", "coordination"],
            "integumentary": ["integumentary", "skin", "wound", "burn", "dermatological", "pressure ulcer", "scar", "healing"],
            "metabolic & endocrine": ["metabolic", "endocrine", "diabetes", "thyroid", "hormone", "glucose", "insulin", "metabolism"],
            "gastrointestinal": ["gastrointestinal", "gi", "digestive", "stomach", "intestine", "bowel", "abdominal"],
            "genitourinary": ["genitourinary", "urinary", "bladder", "kidney", "pelvic", "reproductive"],
            "lymphatic": ["lymphatic", "lymph", "edema", "lymphedema", "swelling", "fluid"],
            "system interactions": ["system interactions", "multisystem", "comorbidity", "interdisciplinary", "complex"],
            "equipment, devices, & technologies": ["equipment", "devices", "technologies", "assistive", "prosthetic", "orthotic", "wheelchair", "walker"],
            "therapeutic modalities": ["therapeutic modalities", "modalities", "ultrasound", "electrical stimulation", "heat", "cold", "traction"],
            "safety & protection": ["safety", "protection", "precautions", "infection control", "fall prevention", "ergonomics"],
            "professional responsibilities": ["professional responsibilities", "ethics", "documentation", "communication", "collaboration", "scope of practice"],
            "research & evidence-based practice": ["research", "evidence-based", "evidence", "clinical practice guidelines", "systematic review", "meta-analysis"]
        }
        relevant_keywords = []
        for topic, keywords in topic_keywords.items():
            if topic in user_topic or any(keyword in user_topic for keyword in keywords):
                relevant_keywords.extend(keywords)
        if not relevant_keywords:
            relevant_keywords = ["physical therapy", "rehabilitation", "treatment", "assessment", "intervention"]
        response_lower = text.lower()
        has_topic_relevance = any(keyword in response_lower for keyword in relevant_keywords)
        
        # References relevance
        has_relevant_references = True
        if "References:" in text:
            refs_section_full = text[text.find("References:"):]
            refs_section = refs_section_full.lower()
            has_actual_urls = any(p in refs_section for p in ["http://", "https://", "www."])
            has_local_pdf_citations = (".pdf" in refs_section) or ("page" in refs_section)
            refs_relevant = any(keyword in refs_section for keyword in relevant_keywords)
            has_irrelevant_refs = any(
                phrase in refs_section for phrase in [
                    "general", "overview", "introduction", "basics", "fundamentals",
                    "what is", "definition", "meaning", "concept"
                ]
            )
            has_relevant_references = (has_actual_urls or has_local_pdf_citations) and refs_relevant and not has_irrelevant_refs
        
        decision = (
            "Y" if (
                has_mcq_format and mcq_before_refs and has_substance and
                not starts_with_resources and
                has_topic_relevance and has_relevant_references
            ) else "N"
        )
        flags = {
            "has_mcq_format": has_mcq_format,
            "mcq_before_refs": mcq_before_refs,
            "has_substance": has_substance,
            "starts_with_resources": starts_with_resources,
            "has_topic_relevance": has_topic_relevance,
            "has_relevant_references": has_relevant_references,
        }
        return decision, flags

    decision, flags = evaluate_helpfulness(response_text)
    has_mcq_format = flags["has_mcq_format"]
    starts_with_resources = flags["starts_with_resources"]
    has_relevant_references = flags["has_relevant_references"]

    if decision == "Y":
        return {
            "messages": [AIMessage(content=f"HELPFULNESS:{decision}")],
            "loop_count": loop_count
        }

    # Provide more specific revision guidance based on what's wrong
    if not has_mcq_format:
        revision_instruction = (
            "You must create an NPTE-style multiple choice question with exactly four choices labeled A., B., C., D. "
            "Do not provide links or general information. Create a clinical scenario or direct question first, then the four answer choices. "
            "Topic: " + initial_query.content
        )
    elif starts_with_resources:
        revision_instruction = (
            "You provided links or general information before the MCQ. You must FIRST create the NPTE-style MCQ (scenario + A-D choices), "
            "then add ONLY relevant references at the bottom under 'References:'. Do not include any content before the MCQ. "
            "Topic: " + initial_query.content
        )
    elif not has_relevant_references:
        revision_instruction = (
            "Your references are not relevant to the topic or are not actual URLs. Include ONLY actual clickable URLs (starting with http:// or https://) "
            "that directly support the MCQ content about " + initial_query.content + ". Remove general, overview, or unrelated links. "
            "References should be specific to the topic and MCQ, and must be real clickable links. "
            "Topic: " + initial_query.content
        )
    else:
        revision_instruction = (
            "Revise your answer to strictly output an NPTE-style MCQ FIRST (question/scenario + A-D choices). "
            "Do not include links or commentary before the MCQ. If you include sources, append them after the MCQ under 'References:'. "
            "Use exactly four choices labeled A., B., C., D., with only one correct answer. Topic: " + initial_query.content
        )

    return {
        "messages": [
            AIMessage(content="HELPFULNESS:N"),
            HumanMessage(content=revision_instruction),
        ],
        "loop_count": loop_count,
    }


def helpfulness_decision(state: Dict[str, Any]):
    """Terminate on 'HELPFULNESS:Y' or loop otherwise; guard against infinite loops."""
    # Check loop-limit marker
    if any(getattr(m, "content", "") == "HELPFULNESS:END" for m in state["messages"][-1:]):
        return END

    last = state["messages"][-1]
    text = getattr(last, "content", "")
    if "HELPFULNESS:Y" in text:
        return "end"
    return "continue"


def build_agent_graph_with_helpfulness(model, system_instruction, format_instruction, checkpointer=None):
    """Build an agent graph with an auxiliary helpfulness evaluation subgraph."""
    from app.tools import get_tool_belt
    from app.agent import ResponseFormat
    
    # Create model-bound functions
    def _call_model(state: AgentState) -> Dict[str, Any]:
        """Wrapper to pass model to call_model with proper tool binding."""
        # Ensure the model has access to all tools
        model_with_tools = build_model_with_tools(model)
        messages = state["messages"]
        
        # Add NPTE-specific system instruction for topic-based queries
        if len(messages) == 1 and isinstance(messages[0], HumanMessage):
            user_topic = messages[0].content.strip()
            
            # Define valid NPTE topics and their variations
            valid_topics = {
                "cardiovascular & pulmonary": ["cardiovascular", "pulmonary", "cardiac", "heart", "vascular", "circulatory", "respiratory", "lung", "breathing", "ventilation", "oxygenation"],
                "musculoskeletal & nervous": ["musculoskeletal", "nervous", "muscle", "skeletal", "bone", "joint", "orthopedic", "neurological", "nerve", "spinal", "cervical", "thoracic", "lumbar"],
                "neuromuscular": ["neuromuscular", "muscle", "nerve", "motor", "neurological", "motor control", "muscle strength", "coordination"],
                "integumentary": ["integumentary", "skin", "wound", "burn", "dermatological", "pressure ulcer", "scar", "healing"],
                "metabolic & endocrine": ["metabolic", "endocrine", "diabetes", "thyroid", "hormone", "glucose", "insulin", "metabolism"],
                "gastrointestinal": ["gastrointestinal", "gi", "digestive", "stomach", "intestine", "bowel", "abdominal"],
                "genitourinary": ["genitourinary", "urinary", "bladder", "kidney", "pelvic", "reproductive"],
                "lymphatic": ["lymphatic", "lymph", "edema", "lymphedema", "swelling", "fluid"],
                "system interactions": ["system interactions", "multisystem", "comorbidity", "interdisciplinary", "complex"],
                "equipment, devices, & technologies": ["equipment", "devices", "technologies", "assistive", "prosthetic", "orthotic", "wheelchair", "walker"],
                "therapeutic modalities": ["therapeutic modalities", "modalities", "ultrasound", "electrical stimulation", "heat", "cold", "traction"],
                "safety & protection": ["safety", "protection", "precautions", "infection control", "fall prevention", "ergonomics"],
                "professional responsibilities": ["professional responsibilities", "ethics", "documentation", "communication", "collaboration", "scope of practice"],
                "research & evidence-based practice": ["research", "evidence-based", "evidence", "clinical practice guidelines", "systematic review", "meta-analysis"]
            }
            
            # Check if user topic is valid
            is_valid_topic = False
            matched_topic = None
            
            for full_topic, variations in valid_topics.items():
                if (user_topic.lower() == full_topic.lower() or 
                    any(variation.lower() == user_topic.lower() for variation in variations)):
                    is_valid_topic = True
                    matched_topic = full_topic
                    break
            
            if not is_valid_topic:
                # Return error message with available topics
                topic_list = "\n".join([f"- {topic}" for topic in valid_topics.keys()])
                error_message = f"""
I don't recognize "{user_topic}" as a valid NPTE topic. 

Please use one of these available topics:

{topic_list}

You can also use individual terms like "musculoskeletal" or "nervous" instead of "musculoskeletal & nervous".
"""
                return {"messages": [AIMessage(content=error_message)]}
            
            # This is the initial query - add NPTE context with tool usage instructions
            npte_prompt = f"""
You are an expert NPTE (National Physical Therapy Examination) question writer. 
The user has provided a topic: "{user_topic}" (which maps to the NPTE domain: "{matched_topic}")

Your task is to create a high-quality NPTE-style multiple choice question about this topic.

IMPORTANT: You have access to the following tools to research current information:
- Web search (Tavily): For current information and news
- ArXiv search: For academic papers and research
- RAG document retrieval: For NPTE-specific documents and guidelines
- Google Scholar search: For academic research papers
- IJSPT search: For sports physical therapy research

USE THESE TOOLS to research current best practices, guidelines, and evidence-based information about {matched_topic} before creating your question.

CRITICAL REQUIREMENTS:
1. FIRST, use appropriate tools to research current information about {matched_topic}
2. Create a realistic clinical scenario or direct question about {matched_topic}
3. Provide exactly 4 multiple choice answers (A, B, C, D)
4. Ensure the question follows 2023-2025 NPTE standards
5. Make the scenario realistic and clinically relevant
6. Ensure all answer choices are plausible but only one is correct

STRICT FORMATTING RULES - FOLLOW EXACTLY:
- FIRST output the question/scenario and the four answer choices (A-D). 
- NO links, lists, or commentary before the MCQ.
- NO general information or resource lists before the MCQ.
- ONLY after the MCQ, add a "References:" section with clickable links that are SPECIFICALLY relevant to answering the MCQ.
- Do NOT reveal which option is correct.
- Do NOT include unrelated or general links.
- References MUST be either actual clickable URLs (starting with http:// or https://) OR citations to local RAG PDFs (e.g., filename.pdf, page N).

REQUIRED FORMAT:
[Clinical Scenario or Direct Question]
A. [Answer choice]
B. [Answer choice] 
C. [Answer choice]
D. [Answer choice]

References:
[ONLY include items that directly support the MCQ content, one per line. Each item MUST be either (a) a clickable URL starting with http(s):// or (b) a local PDF citation in the form 'filename.pdf (page N)'. Do not include plain text titles without a link or PDF citation.]

Start by researching the topic using the available tools, then create the MCQ with relevant references.
"""
            # Create a new message with the enhanced prompt
            enhanced_message = HumanMessage(content=npte_prompt)
            messages = [enhanced_message]
        
        # Invoke the model with tools
        response = model_with_tools.invoke(messages)
        if isinstance(response, AIMessage) and not getattr(response, "tool_calls", None):
            content = getattr(response, "content", "")
            if "References:" in content:
                try:
                    fallback_query = state["messages"][0].content if state.get("messages") else None
                except Exception:
                    fallback_query = None
                filtered = _filter_references(content, fallback_query)
                response = AIMessage(content=filtered)
            else:
                try:
                    topic = state["messages"][0].content if state.get("messages") else ""
                except Exception:
                    topic = ""
                fallback = f"https://scholar.google.com/scholar?q={urllib.parse.quote('NPTE ' + topic)}"
                response = AIMessage(content=f"{content.rstrip()}\n\nReferences:\n{fallback}")
        return {"messages": [response]}
    
    def _helpfulness_node(state: AgentState) -> Dict[str, Any]:
        """Wrapper to pass model to helpfulness_node."""
        return helpfulness_node(state, model)
    
    graph = StateGraph(AgentState)
    tool_node = ToolNode(get_tool_belt())
    
    graph.add_node("agent", _call_model)
    graph.add_node("action", tool_node)
    graph.add_node("helpfulness", _helpfulness_node)
    graph.set_entry_point("agent")
    
    graph.add_conditional_edges(
        "agent",
        route_to_action_or_helpfulness,
        {"action": "action", "helpfulness": "helpfulness"},
    )
    graph.add_conditional_edges(
        "helpfulness",
        helpfulness_decision,
        {"continue": "agent", "end": END, END: END},
    )
    graph.add_edge("action", "agent")
    
    return graph.compile(checkpointer=checkpointer)


def graph():
    """Zero-argument graph builder for LangGraph server."""
    import os
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import MemorySaver

    model = ChatOpenAI(
        model=os.getenv('TOOL_LLM_NAME', 'gpt-4o-mini'),
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('TOOL_LLM_URL', 'https://api.openai.com/v1'),
        temperature=0,
    )
    system_instruction = (
        'You are an expert NPTE (National Physical Therapy Examination) question writer.'
    )
    format_instruction = (
        'Respond with NPTE-style multiple choice questions using exactly four options (A, B, C, D).'
    )
    return build_agent_graph_with_helpfulness(
        model,
        system_instruction,
        format_instruction,
        checkpointer=MemorySaver(),
    )
