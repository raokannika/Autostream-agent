import os
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from rag import retrieve
from tools import mock_lead_capture

load_dotenv()

INTENT_SYSTEM_PROMPT = """You are an intent classifier for AutoStream, a SaaS video editing platform.

Classify the user's latest message into exactly ONE of these intents:
- GREETING: Casual greetings, hellos, small talk
- PRODUCT_INQUIRY: Questions about features, pricing, plans, policies, or how the product works
- HIGH_INTENT: User is expressing readiness to sign up, buy, or try a specific plan

Respond with ONLY the intent label, nothing else. Example: HIGH_INTENT"""

AGENT_SYSTEM_PROMPT = """You are Alex, a friendly and knowledgeable sales assistant for AutoStream — an AI-powered video editing SaaS platform for content creators.

Your personality:
- Professional but conversational
- Enthusiastic about helping creators grow
- Concise — no unnecessary filler

AutoStream Plans:
- Basic Plan: $29/month — 10 videos/month, 720p resolution, email support
- Pro Plan: $79/month — Unlimited videos, 4K resolution, AI captions, 24/7 support

Policies:
- No refunds after 7 days
- 24/7 support only on Pro plan

When answering product questions, use the retrieved knowledge provided.
When the user shows high interest in signing up, warmly acknowledge it and begin collecting their details one at a time:
1. First ask for their name
2. Then ask for their email
3. Then ask which platform they create on (YouTube, Instagram, TikTok, etc.)

Do NOT call the lead capture tool until you have all three: name, email, and platform.
Never ask for multiple pieces of information in a single message."""


class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    intent: Optional[str]
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_captured: bool
    collecting_lead: bool
    awaiting_field: Optional[str]


def make_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Copy .env.example to .env and add your Groq key."
        )
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=api_key,
        temperature=0.3,
    )


def classify_intent(state: AgentState, llm: ChatGroq) -> AgentState:
    messages = state["messages"]
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
    )
    if not last_human:
        return {**state, "intent": "GREETING"}

    response = llm.invoke([
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=last_human.content)
    ])
    intent = response.content.strip().upper()
    if intent not in ["GREETING", "PRODUCT_INQUIRY", "HIGH_INTENT"]:
        intent = "PRODUCT_INQUIRY"
    return {**state, "intent": intent}


def retrieve_knowledge(state: AgentState) -> AgentState:
    messages = state["messages"]
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
    )
    if not last_human:
        return state
    context = retrieve(last_human.content)
    enriched = list(messages)
    enriched.append(SystemMessage(content=f"Retrieved knowledge:\n{context}"))
    return {**state, "messages": enriched}


def extract_field_value(user_text: str, field: str, llm: ChatGroq) -> str:
    extraction_prompt = f"""Extract the {field} from the following user message.
Return ONLY the extracted value, nothing else. If not found, return NONE.

User message: {user_text}"""
    response = llm.invoke([HumanMessage(content=extraction_prompt)])
    value = response.content.strip()
    if value.upper() == "NONE" or not value:
        return ""
    return value


def handle_lead_collection(state: AgentState, llm: ChatGroq) -> AgentState:
    messages = state["messages"]
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
    )
    user_text = last_human.content if last_human else ""

    awaiting = state.get("awaiting_field")
    lead_name = state.get("lead_name")
    lead_email = state.get("lead_email")
    lead_platform = state.get("lead_platform")

    if awaiting == "name" and not lead_name:
        lead_name = extract_field_value(user_text, "person's name", llm)

    elif awaiting == "email" and not lead_email:
        lead_email = extract_field_value(user_text, "email address", llm)

    elif awaiting == "platform" and not lead_platform:
        lead_platform = extract_field_value(user_text, "social media or video platform", llm)

    if lead_name and lead_email and lead_platform:
        mock_lead_capture(lead_name, lead_email, lead_platform)
        confirmation = (
            f"You're all set, {lead_name}! 🎉\n\n"
            f"I've registered your interest in AutoStream's Pro Plan.\n"
            f"A member of our team will reach out to **{lead_email}** shortly.\n\n"
            f"Welcome to AutoStream — can't wait to see what you create on {lead_platform}!"
        )
        return {
            **state,
            "lead_name": lead_name,
            "lead_email": lead_email,
            "lead_platform": lead_platform,
            "lead_captured": True,
            "collecting_lead": False,
            "awaiting_field": None,
            "messages": list(messages) + [AIMessage(content=confirmation)]
        }

    if not lead_name:
        next_field = "name"
        question = "Awesome! I'd love to get you started. What's your name?"
    elif not lead_email:
        next_field = "email"
        question = f"Great, {lead_name}! What's the best email address to reach you at?"
    else:
        next_field = "platform"
        question = "Almost there! Which platform do you mainly create content on? (e.g. YouTube, Instagram, TikTok)"

    return {
        **state,
        "lead_name": lead_name,
        "lead_email": lead_email,
        "lead_platform": lead_platform,
        "collecting_lead": True,
        "awaiting_field": next_field,
        "messages": list(messages) + [AIMessage(content=question)]
    }


def generate_response(state: AgentState, llm: ChatGroq) -> AgentState:
    messages = state["messages"]
    system = SystemMessage(content=AGENT_SYSTEM_PROMPT)
    history = [system] + [
        m for m in messages
        if isinstance(m, (HumanMessage, AIMessage, SystemMessage))
        and not (isinstance(m, SystemMessage) and m.content == AGENT_SYSTEM_PROMPT)
    ]
    response = llm.invoke(history)
    return {**state, "messages": list(messages) + [AIMessage(content=response.content)]}


def build_graph(llm: ChatGroq) -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("classify", lambda s: classify_intent(s, llm))
    graph.add_node("retrieve", retrieve_knowledge)
    graph.add_node("collect_lead", lambda s: handle_lead_collection(s, llm))
    graph.add_node("respond", lambda s: generate_response(s, llm))

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify",
        route_after_intent,
        {
            "retrieve": "retrieve",
            "collect_lead": "collect_lead",
            "respond": "respond",
            "end": END,
        }
    )

    graph.add_edge("retrieve", "respond")
    graph.add_edge("respond", END)
    graph.add_edge("collect_lead", END)

    return graph.compile()


def route_after_intent(state: AgentState) -> str:
    if state.get("lead_captured"):
        return "end"
    if state.get("collecting_lead"):
        return "collect_lead"
    intent = state.get("intent", "GREETING")
    if intent == "HIGH_INTENT":
        return "collect_lead"
    if intent == "PRODUCT_INQUIRY":
        return "retrieve"
    return "respond"


def initialize_state() -> AgentState:
    return {
        "messages": [],
        "intent": None,
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False,
        "collecting_lead": False,
        "awaiting_field": None,
    }


def run_agent():
    print("\n" + "="*60)
    print("  AutoStream AI Sales Assistant")
    print("  Powered by LangGraph + Groq (Llama 3.1 8B)")
    print("="*60)
    print("  Type 'exit' or 'quit' to end the session.\n")

    llm = make_llm()
    graph = build_graph(llm)
    state = initialize_state()

    opening = AIMessage(content=(
        "Hey there! 👋 Welcome to AutoStream — your AI-powered video editing platform.\n"
        "I'm Alex, your assistant. Whether you have questions about our plans, features, "
        "or you're ready to get started, I'm here to help. What's on your mind?"
    ))
    state["messages"].append(opening)
    print(f"Alex: {opening.content}\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            print("\nAlex: Thanks for stopping by! Feel free to come back anytime. 👋")
            break

        state["messages"].append(HumanMessage(content=user_input))
        state = graph.invoke(state)

        last_ai = next(
            (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None
        )
        if last_ai:
            print(f"\nAlex: {last_ai.content}\n")

        if state.get("lead_captured"):
            print("[ Session complete — lead successfully captured. ]\n")
            break


if __name__ == "__main__":
    run_agent()
