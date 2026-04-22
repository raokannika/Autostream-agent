# AutoStream AI Sales Agent

A conversational AI agent built for AutoStream — a fictional SaaS video editing platform. Built using **LangGraph**, **LangChain**, and **Groq (Llama 3.1 8B)**. The agent identifies user intent, retrieves product knowledge via RAG, and captures leads through a structured multi-turn conversation.

---

## How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/raokannika/autostream-agent.git
cd autostream-agent
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy the example file and add your API key:

```bash
cp .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at [https://console.groq.com/keys](https://console.groq.com/keys)

### 4. Run the Agent

```bash
python agent.py
```

---

## Project Structure

```
autostream-agent/
├── agent.py            # LangGraph agent — intent routing, state machine, response generation
├── rag.py              # RAG retrieval pipeline — keyword-scored document retrieval
├── tools.py            # Mock lead capture tool
├── knowledge_base.json # Local knowledge base — pricing, policies, FAQs
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── README.md           # This file
└── ABOUT.md            # Assignment context
```

---

## Architecture Explanation

### Why LangGraph?

LangGraph was chosen over AutoGen or vanilla LangChain chains because the lead qualification workflow is inherently **stateful and branching**. At each conversation turn, the agent must make routing decisions based on:

- The current intent (Greeting / Product Inquiry / High Intent)
- Whether lead collection has started
- Which lead field is still missing (name → email → platform)

LangGraph's `StateGraph` models this as an explicit directed graph with typed state transitions, making the logic traceable, testable, and easy to extend. Each node is a pure function that reads from and writes to a shared `AgentState` TypedDict.

### State Management

State is maintained in a persistent `AgentState` dictionary across all conversation turns. It tracks:

| Field | Purpose |
|---|---|
| `messages` | Full conversation history (used for LLM context and intent classification) |
| `intent` | Latest classified intent: `GREETING`, `PRODUCT_INQUIRY`, `HIGH_INTENT` |
| `collecting_lead` | Flag indicating lead capture has started |
| `awaiting_field` | Which field the agent is currently waiting for (`name`, `email`, `platform`) |
| `lead_name/email/platform` | Collected lead details |
| `lead_captured` | Set to `True` when mock tool has been called — terminates the flow |

This design ensures the agent can handle interruptions (e.g., user asking a product question mid-lead-flow) without losing context, and the `mock_lead_capture()` tool is never called until all three values are confirmed present.

### RAG Pipeline

The knowledge base (`knowledge_base.json`) contains pricing plans, company policies, and FAQs. The `rag.py` module converts these into flat document strings and performs keyword-overlap scoring against the user query — a zero-dependency, fast retrieval approach suitable for this scale. The top-3 results are injected as a `SystemMessage` before LLM response generation.

### Graph Flow

```
User Input
    │
    ▼
[classify] ──── GREETING ─────────────────────► [respond] ──► END
    │
    ├── PRODUCT_INQUIRY ──► [retrieve] ──────► [respond] ──► END
    │
    ├── HIGH_INTENT ──────► [collect_lead] ──► END (loop until all fields collected)
    │
    └── (already collecting) ► [collect_lead] ──► END
```

---

## WhatsApp Integration via Webhooks

To deploy this agent on WhatsApp:

1. **Register a WhatsApp Business API** account through Meta (or a BSP like Twilio).

2. **Set up a webhook endpoint** — a FastAPI or Flask server that receives `POST` requests from WhatsApp whenever a user sends a message. Meta sends a JSON payload containing the user's phone number, message body, and metadata.

3. **Map phone number → agent state**: Each WhatsApp user gets a persistent `AgentState` keyed by their phone number (stored in Redis or a database). On every incoming webhook, load the state, append the new `HumanMessage`, run the LangGraph, and persist the updated state.

4. **Send the response**: Use the WhatsApp Cloud API (`POST /messages`) with the agent's reply text.

5. **Webhook verification**: During setup, Meta sends a `GET` request with a `hub.challenge` token. Your server must return this token to confirm ownership.

**Example FastAPI skeleton:**

```python
from fastapi import FastAPI, Request
from agent import build_graph, AgentState
import redis, json

app = FastAPI()
r = redis.Redis()
graph = build_graph()

@app.post("/webhook")
async def receive_message(request: Request):
    body = await request.json()
    phone = body["entry"][0]["changes"][0]["value"]["contacts"][0]["wa_id"]
    text = body["entry"][0]["changes"][0]["value"]["messages"][0]["text"]["body"]
    raw = r.get(phone)
    state = json.loads(raw) if raw else initialize_state()
    state["messages"].append({"role": "human", "content": text})
    state = graph.invoke(state)
    r.set(phone, json.dumps(state))
    reply = [m for m in reversed(state["messages"]) if m["role"] == "ai"][0]["content"]
    send_whatsapp_message(phone, reply)
    return {"status": "ok"}
```

This approach scales horizontally since state lives in Redis rather than in-process memory.

---

## Example Conversation

```
Alex: Hey there! Welcome to AutoStream...

You: Hi, tell me about your pricing.

Alex: Great question! AutoStream offers two plans:
- Basic Plan ($29/month): 10 videos/month, 720p resolution, email support.
- Pro Plan ($79/month): Unlimited videos, 4K resolution, AI captions, 24/7 support.

You: That sounds great, I want to try the Pro plan for my YouTube channel.

Alex: Awesome! I'd love to get you started. What's your name?

You: Sarah

Alex: Great, Sarah! What's the best email address to reach you at?

You: sarah@example.com

Alex: Almost there! Which platform do you mainly create content on?

You: YouTube

Alex: You're all set, Sarah! 🎉 I've registered your interest in AutoStream's Pro Plan...

==================================================
  LEAD CAPTURED SUCCESSFULLY
==================================================
  Name     : Sarah
  Email    : sarah@example.com
  Platform : YouTube
==================================================
```

---

## Evaluation Notes

| Criterion | Implementation |
|---|---|
| Intent Detection | Dedicated LLM classification node with strict prompt and 3-class output |
| RAG | Keyword-scored retrieval over JSON knowledge base, injected as context |
| State Management | LangGraph `AgentState` TypedDict persisted across all turns |
| Tool Calling | `mock_lead_capture()` called only after name + email + platform confirmed |
| Code Clarity | Single-responsibility nodes, typed state, no inline magic |
| Deployability | Stateless graph + external state store = horizontally scalable |
