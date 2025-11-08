# Talkshow & Conversational Multi-Agent Implementation Guide

**Framework**: LangGraph
**Use Case**: Debates, panel discussions, interviews, group conversations
**Last Updated**: 2025-11-08

---

## Table of Contents

1. [Overview](#overview)
2. [Complete Talkshow Implementation](#complete-talkshow-implementation)
3. [Debate System Implementation](#debate-system-implementation)
4. [Panel Discussion Implementation](#panel-discussion-implementation)
5. [Dynamic GroupChat Implementation](#dynamic-groupchat-implementation)
6. [Advanced Features](#advanced-features)
7. [Production Considerations](#production-considerations)

---

## Overview

Talkshow and conversational multi-agent systems benefit from LangGraph's:
- **State management**: Track conversation flow, speaking turns, topics
- **Dynamic routing**: Determine next speaker based on context
- **Message accumulation**: Build natural conversation history
- **Flexible control**: Mix structured (moderator-led) and organic (debate) flows

---

## Complete Talkshow Implementation

### Full Working Example

```python
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.types import Command
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing import Literal, Annotated
import operator

# ============================================================================
# STATE DEFINITION
# ============================================================================

class TalkshowState(MessagesState):
    """Complete state for talkshow management"""

    # Conversation metadata
    topic: str
    episode_number: int

    # Participants
    host_name: str
    guest_names: list[str]

    # Flow control
    current_speaker: str  # Who is speaking now
    speaking_turn: int  # Overall turn counter
    questions_asked: int  # Host question counter

    # Episode configuration
    max_questions: int  # Total questions for episode
    questions_per_guest: dict[str, int]  # Track per guest

    # Conversation quality
    topic_coverage: list[str]  # Topics discussed
    guest_participation: dict[str, int]  # Speaking turns per guest


# ============================================================================
# AGENT IMPLEMENTATIONS
# ============================================================================

def host_agent(state: TalkshowState) -> Command[Literal["guest_1", "guest_2", "guest_3", "end_show"]]:
    """
    Host manages show flow:
    - Opens show
    - Asks contextual questions
    - Manages transitions between guests
    - Closes show
    """

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    # Determine current phase
    if state.speaking_turn == 0:
        # === OPENING ===
        opening_prompt = f"""
        You are the host of a talkshow.

        Episode #{state.episode_number}
        Topic: {state.topic}
        Guests: {', '.join(state.guest_names)}

        Provide an engaging opening (2-3 sentences):
        1. Welcome the audience
        2. Introduce the topic
        3. Build excitement
        """

        opening = llm.invoke(opening_prompt).content

        return Command(
            goto="guest_1",  # Start with first guest
            update={
                "messages": [AIMessage(content=opening, name=state.host_name)],
                "speaking_turn": state.speaking_turn + 1,
                "current_speaker": "guest_1"
            }
        )

    elif state.questions_asked >= state.max_questions:
        # === CLOSING ===
        closing_prompt = f"""
        You are concluding a talkshow episode.

        Topic discussed: {state.topic}
        Guests: {', '.join(state.guest_names)}

        Key topics covered:
        {chr(10).join(f"- {topic}" for topic in state.topic_coverage)}

        Provide closing remarks (2-3 sentences):
        1. Summarize the discussion
        2. Thank the guests
        3. Sign off warmly
        """

        closing = llm.invoke(closing_prompt).content

        return Command(
            goto="end_show",
            update={
                "messages": [AIMessage(content=closing, name=state.host_name)],
                "speaking_turn": state.speaking_turn + 1
            }
        )

    else:
        # === ASK QUESTION ===

        # Determine which guest to ask
        # Balance questions between guests
        guest_idx = state.questions_asked % len(state.guest_names)
        target_guest = state.guest_names[guest_idx]
        target_guest_node = f"guest_{guest_idx + 1}"

        # Generate contextual question based on conversation
        question_prompt = f"""
        You are a talkshow host. Conversation context:

        Topic: {state.topic}
        Current guest: {target_guest}

        Recent conversation (last 3 exchanges):
        {format_recent_messages(state.messages, n=6)}

        Topics already covered:
        {', '.join(state.topic_coverage) if state.topic_coverage else 'None yet'}

        Generate an insightful question for {target_guest}:
        - Build on previous answers if relevant
        - Explore new angles of {state.topic}
        - Be specific and thought-provoking
        - Keep it conversational (1-2 sentences)
        """

        question = llm.invoke(question_prompt).content

        # Extract topic for tracking
        topic_mentioned = extract_topic(question, state.topic)
        if topic_mentioned and topic_mentioned not in state.topic_coverage:
            updated_coverage = state.topic_coverage + [topic_mentioned]
        else:
            updated_coverage = state.topic_coverage

        return Command(
            goto=target_guest_node,
            update={
                "messages": [AIMessage(content=question, name=state.host_name)],
                "speaking_turn": state.speaking_turn + 1,
                "questions_asked": state.questions_asked + 1,
                "current_speaker": target_guest_node,
                "topic_coverage": updated_coverage
            }
        )


def create_guest_agent(guest_name: str, expertise: str, personality: str):
    """
    Factory function to create guest agents with specific characteristics
    """
    def guest_agent(state: TalkshowState) -> Command[Literal["host"]]:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o", temperature=0.8)  # Higher temp for personality

        # Get the question just asked
        last_message = state.messages[-1]

        # Build rich context for guest
        response_prompt = f"""
        You are {guest_name}, a guest on a talkshow.

        Your expertise: {expertise}
        Your personality: {personality}

        Episode topic: {state.topic}

        Conversation so far (recent):
        {format_recent_messages(state.messages, n=4)}

        The host just asked: "{last_message.content}"

        Provide your response:
        - Draw on your expertise: {expertise}
        - Show your personality: {personality}
        - Be conversational and engaging (2-4 sentences)
        - Reference previous points if relevant
        - Add insights, examples, or anecdotes
        """

        response = llm.invoke(response_prompt).content

        # Track participation
        updated_participation = state.guest_participation.copy()
        updated_participation[guest_name] = updated_participation.get(guest_name, 0) + 1

        # Always hand back to host
        return Command(
            goto="host",
            update={
                "messages": [AIMessage(content=response, name=guest_name)],
                "speaking_turn": state.speaking_turn + 1,
                "current_speaker": state.host_name,
                "guest_participation": updated_participation
            }
        )

    return guest_agent


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_recent_messages(messages: list, n: int = 6) -> str:
    """Format last N messages for context"""
    recent = messages[-n:] if len(messages) >= n else messages

    formatted = []
    for msg in recent:
        speaker = msg.name if hasattr(msg, 'name') else "Unknown"
        content = msg.content
        formatted.append(f"{speaker}: {content}")

    return "\n".join(formatted)


def extract_topic(question: str, main_topic: str) -> str:
    """Extract specific sub-topic from question"""
    # Simple heuristic - in production, use LLM
    keywords = question.lower().split()

    # Common subtopic patterns for AI governance example
    if "regulation" in keywords or "policy" in keywords:
        return "regulation"
    elif "ethics" in keywords or "bias" in keywords:
        return "ethics"
    elif "safety" in keywords or "risk" in keywords:
        return "safety"
    elif "innovation" in keywords or "progress" in keywords:
        return "innovation"
    else:
        return None


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def build_talkshow_graph():
    """Construct the complete talkshow graph"""

    graph = StateGraph(TalkshowState)

    # Add host node
    graph.add_node("host", host_agent)

    # Add guest nodes
    graph.add_node("guest_1", create_guest_agent(
        guest_name="Dr. Sarah Chen",
        expertise="AI Ethics and Policy",
        personality="Thoughtful, measured, emphasizes ethical considerations"
    ))

    graph.add_node("guest_2", create_guest_agent(
        guest_name="Prof. Marcus Williams",
        expertise="Machine Learning Research",
        personality="Enthusiastic, technical, optimistic about AI potential"
    ))

    graph.add_node("guest_3", create_guest_agent(
        guest_name="Alex Rivera",
        expertise="AI Industry and Startups",
        personality="Pragmatic, business-focused, shares real-world examples"
    ))

    # Add end node
    def end_show(state: TalkshowState):
        """Final node - compile episode summary"""
        return {
            "messages": [AIMessage(
                content=f"[Episode ended. {state.questions_asked} questions asked, "
                        f"{len(state.topic_coverage)} topics covered]",
                name="system"
            )]
        }

    graph.add_node("end_show", end_show)

    # Set entry point
    graph.set_entry_point("host")

    # Add terminal edge
    graph.add_edge("end_show", END)

    return graph


# ============================================================================
# USAGE
# ============================================================================

def run_talkshow_episode(topic: str, max_questions: int = 6):
    """Run a complete talkshow episode"""

    # Build graph
    graph = build_talkshow_graph()
    app = graph.compile()

    # Initialize state
    initial_state = {
        "topic": topic,
        "episode_number": 1,
        "host_name": "Jamie Anderson",
        "guest_names": ["Dr. Sarah Chen", "Prof. Marcus Williams", "Alex Rivera"],
        "current_speaker": "host",
        "speaking_turn": 0,
        "questions_asked": 0,
        "max_questions": max_questions,
        "questions_per_guest": {},
        "topic_coverage": [],
        "guest_participation": {},
        "messages": []
    }

    # Run episode
    result = app.invoke(initial_state)

    # Display conversation
    print(f"\n{'='*80}")
    print(f"TALKSHOW EPISODE: {topic}")
    print(f"{'='*80}\n")

    for message in result["messages"]:
        speaker = message.name if hasattr(message, 'name') else "Unknown"
        content = message.content
        print(f"{speaker}:")
        print(f"  {content}\n")

    # Summary stats
    print(f"\n{'='*80}")
    print("EPISODE STATISTICS")
    print(f"{'='*80}")
    print(f"Total speaking turns: {result['speaking_turn']}")
    print(f"Questions asked: {result['questions_asked']}")
    print(f"Topics covered: {', '.join(result['topic_coverage'])}")
    print(f"\nGuest participation:")
    for guest, count in result['guest_participation'].items():
        print(f"  {guest}: {count} responses")
    print()

    return result


# Example usage
if __name__ == "__main__":
    run_talkshow_episode(
        topic="The Future of AI Governance",
        max_questions=6  # 2 questions per guest
    )
```

---

## Debate System Implementation

### Structured Debate with Scoring

```python
from langgraph.graph import StateGraph, MessagesState, END
from typing import Literal
from langchain_core.messages import AIMessage

class DebateState(MessagesState):
    """State for formal debate"""
    topic: str
    position_for: str  # Supporting position
    position_against: str  # Opposing position

    # Debate structure
    current_round: int
    max_rounds: int
    current_speaker: Literal["moderator", "for", "against", "judge"]

    # Scoring
    scores_for: list[float]  # Per-round scores
    scores_against: list[float]
    judging_criteria: list[str]

def moderator(state: DebateState):
    """Introduces and manages debate"""
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o")

    if state.current_round == 0:
        # Opening
        intro = f"""
        Welcome to tonight's debate!

        Topic: {state.topic}

        For the motion: {state.position_for}
        Against the motion: {state.position_against}

        This will be a {state.max_rounds}-round debate.
        Judging criteria: {', '.join(state.judging_criteria)}

        Let's begin with opening statements.
        """
        return {
            "messages": [AIMessage(content=intro, name="Moderator")],
            "current_speaker": "for"
        }

    elif state.current_round > state.max_rounds:
        # Transition to judging
        transition = "Both debaters have presented their cases. Let's hear the judge's verdict."
        return {
            "messages": [AIMessage(content=transition, name="Moderator")],
            "current_speaker": "judge"
        }

    else:
        # Round transition
        transition = f"Round {state.current_round} complete. Let's continue."
        return {
            "messages": [AIMessage(content=transition, name="Moderator")],
            "current_speaker": "for"
        }

def debater_for(state: DebateState):
    """Argues FOR the motion"""
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    round_type = "opening statement" if state.current_round == 1 else "rebuttal and argument"

    prompt = f"""
    You are debating FOR: {state.topic}
    Position: {state.position_for}

    Round {state.current_round} - {round_type}

    Previous arguments:
    {format_recent_messages(state.messages, n=4)}

    Provide your {round_type}:
    - If opening: Present your strongest arguments (3-4 sentences)
    - If rebuttal: Counter opponent's points and strengthen your position (3-4 sentences)
    - Use logic, evidence, and clear reasoning
    - Address judging criteria: {', '.join(state.judging_criteria)}
    """

    response = llm.invoke(prompt).content

    return {
        "messages": [AIMessage(content=response, name="Debater FOR")],
        "current_speaker": "against"
    }

def debater_against(state: DebateState):
    """Argues AGAINST the motion"""
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    round_type = "opening statement" if state.current_round == 1 else "rebuttal and argument"

    prompt = f"""
    You are debating AGAINST: {state.topic}
    Position: {state.position_against}

    Round {state.current_round} - {round_type}

    Previous arguments:
    {format_recent_messages(state.messages, n=4)}

    Provide your {round_type}:
    - If opening: Present your strongest counter-arguments (3-4 sentences)
    - If rebuttal: Refute opponent's points and defend your position (3-4 sentences)
    - Use logic, evidence, and clear reasoning
    - Address judging criteria: {', '.join(state.judging_criteria)}
    """

    response = llm.invoke(prompt).content

    return {
        "messages": [AIMessage(content=response, name="Debater AGAINST")],
        "current_speaker": "moderator",
        "current_round": state.current_round + 1  # Increment round
    }

def judge(state: DebateState):
    """Evaluates debate and provides verdict"""
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)  # Lower temp for objectivity

    prompt = f"""
    You are judging a formal debate.

    Topic: {state.topic}
    FOR: {state.position_for}
    AGAINST: {state.position_against}

    Judging criteria:
    {chr(10).join(f"- {c}" for c in state.judging_criteria)}

    Full debate transcript:
    {format_all_messages(state.messages)}

    Provide your judgment:
    1. Score each side on each criterion (1-10)
    2. Identify strongest arguments from each side
    3. Declare winner with clear reasoning
    4. Total word count: 200-300 words

    Format:
    **Scores:**
    FOR: [criterion]: X/10, [criterion]: X/10, Total: X
    AGAINST: [criterion]: X/10, [criterion]: X/10, Total: X

    **Analysis:**
    [Your detailed analysis]

    **Winner:** [FOR/AGAINST] because [reasoning]
    """

    judgment = llm.invoke(prompt).content

    return {
        "messages": [AIMessage(content=judgment, name="Judge")],
        "current_speaker": "end"
    }

def debate_router(state: DebateState) -> Literal["moderator", "for", "against", "judge", END]:
    """Route to next speaker"""
    speaker = state.current_speaker

    if speaker == "end":
        return END
    elif speaker == "moderator":
        if state.current_round > state.max_rounds:
            return "judge"
        else:
            return "for"
    elif speaker == "for":
        return "against"
    elif speaker == "against":
        return "moderator"
    elif speaker == "judge":
        return END
    else:
        return "moderator"

# Build debate graph
def build_debate_graph():
    graph = StateGraph(DebateState)

    graph.add_node("moderator", moderator)
    graph.add_node("for", debater_for)
    graph.add_node("against", debater_against)
    graph.add_node("judge", judge)

    graph.set_entry_point("moderator")

    # Add routing
    for node in ["moderator", "for", "against", "judge"]:
        graph.add_conditional_edges(node, debate_router)

    return graph.compile()

# Usage
def run_debate(topic: str, rounds: int = 3):
    app = build_debate_graph()

    result = app.invoke({
        "topic": topic,
        "position_for": "Supporting the motion",
        "position_against": "Opposing the motion",
        "current_round": 0,
        "max_rounds": rounds,
        "current_speaker": "moderator",
        "scores_for": [],
        "scores_against": [],
        "judging_criteria": [
            "Argument strength",
            "Evidence quality",
            "Rebuttal effectiveness",
            "Logical coherence"
        ],
        "messages": []
    })

    return result
```

---

## Panel Discussion Implementation

### LLM-Routed Organic Conversation

```python
class PanelState(MessagesState):
    topic: str
    panelists: list[str]  # List of panelist names
    speaking_history: list[str]  # Track who spoke when
    max_turns: int
    current_turn: int

def llm_panel_router(state: PanelState) -> str:
    """Use LLM to dynamically decide next speaker"""
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)  # Fast, cheap for routing

    # Moderator intervenes every 5 turns
    if state.current_turn % 5 == 0 and state.current_turn > 0:
        return "moderator"

    # Get recent speakers
    recent_speakers = state.speaking_history[-3:] if len(state.speaking_history) >= 3 else state.speaking_history

    routing_prompt = f"""
    Panel discussion on: {state.topic}
    Panelists: {', '.join(state.panelists)}

    Recent speakers: {' → '.join(recent_speakers)}

    Last 3 statements:
    {format_recent_messages(state.messages, n=3)}

    Who should speak next? Consider:
    1. Who hasn't spoken recently
    2. Who has relevant expertise for current point
    3. Who would provide interesting counterpoint or build on the idea

    Respond with ONLY the panelist name, nothing else.
    """

    next_speaker = llm.invoke(routing_prompt).content.strip()

    # Validate and fallback
    if next_speaker not in state.panelists + ["moderator"]:
        # Fallback: pick least recent speaker
        for panelist in state.panelists:
            if panelist not in recent_speakers:
                return panelist
        # If all spoke recently, use first panelist
        return state.panelists[0]

    return next_speaker

def create_panelist(name: str, background: str, viewpoint: str):
    def panelist(state: PanelState):
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o", temperature=0.8)

        prompt = f"""
        You are {name}, a panelist in a discussion.

        Your background: {background}
        Your viewpoint: {viewpoint}

        Topic: {state.topic}

        Recent discussion:
        {format_recent_messages(state.messages, n=5)}

        Contribute to the discussion (2-3 sentences):
        - React to previous points (agree, disagree, build upon)
        - Offer your unique perspective based on {background}
        - Be conversational and engage with other panelists
        - You can directly address others by name
        """

        response = llm.invoke(prompt).content

        return {
            "messages": [AIMessage(content=response, name=name)],
            "speaking_history": state.speaking_history + [name],
            "current_turn": state.current_turn + 1
        }

    return panelist

# Build panel
def build_panel_discussion():
    graph = StateGraph(PanelState)

    # Add moderator
    def moderator(state: PanelState):
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o")

        if state.current_turn == 0:
            intro = f"Welcome to our panel on {state.topic}. Panelists, please share your thoughts."
            return {"messages": [AIMessage(content=intro, name="Moderator")]}

        # Mid-discussion intervention
        intervention_prompt = f"""
        You're moderating a panel on {state.topic}.

        Recent discussion:
        {format_recent_messages(state.messages, n=6)}

        Provide brief moderation (1-2 sentences):
        - Summarize key points made
        - Ask a new question to explore different angle
        - Encourage quieter panelists if needed
        """

        response = llm.invoke(intervention_prompt).content
        return {"messages": [AIMessage(content=response, name="Moderator")]}

    graph.add_node("moderator", moderator)

    # Add panelists
    graph.add_node("alice", create_panelist(
        "Alice",
        "AI researcher with 10 years in academia",
        "Optimistic about AI but concerned about safety"
    ))
    graph.add_node("bob", create_panelist(
        "Bob",
        "Tech entrepreneur, founded 2 AI startups",
        "Believes in rapid innovation and market forces"
    ))
    graph.add_node("carol", create_panelist(
        "Carol",
        "Policy advisor specializing in tech regulation",
        "Advocates for proactive governance frameworks"
    ))

    graph.set_entry_point("moderator")

    # Dynamic routing
    def route(state: PanelState):
        if state.current_turn >= state.max_turns:
            return END
        return llm_panel_router(state)

    for node in ["moderator", "alice", "bob", "carol"]:
        graph.add_conditional_edges(node, route, {
            "moderator": "moderator",
            "alice": "alice",
            "bob": "bob",
            "carol": "carol",
            END: END
        })

    return graph.compile()
```

---

## Advanced Features

### 1. Streaming for Real-Time Display

```python
async def stream_talkshow(topic: str):
    """Stream talkshow responses in real-time"""
    graph = build_talkshow_graph()
    app = graph.compile()

    initial_state = {
        "topic": topic,
        "max_questions": 4,
        # ... other init values
    }

    async for event in app.astream(initial_state):
        for node_name, node_output in event.items():
            if "messages" in node_output:
                for msg in node_output["messages"]:
                    speaker = msg.name
                    content = msg.content
                    print(f"\n{speaker}: {content}", flush=True)
```

### 2. Human-in-the-Loop Audience Questions

```python
class InteractiveTalkshowState(TalkshowState):
    audience_questions: list[str]
    allow_audience_participation: bool

def host_with_audience(state: InteractiveTalkshowState):
    # Check if audience question pending
    if state.audience_questions and state.questions_asked % 3 == 0:
        # Every 3rd question, take audience question
        audience_q = state.audience_questions.pop(0)

        return Command(
            goto=select_best_guest_for_question(audience_q, state.guest_names),
            update={
                "messages": [AIMessage(
                    content=f"Great question from our audience: {audience_q}",
                    name=state.host_name
                )]
            }
        )
    else:
        # Normal host behavior
        return regular_host_behavior(state)

# Inject audience question during execution
async def interactive_session():
    app = build_interactive_graph()

    config = {"configurable": {"thread_id": "show-123"}}

    # Start show
    await app.ainvoke(initial_state, config)

    # Pause for audience question
    current_state = app.get_state(config)

    # Add audience question
    current_state.values["audience_questions"].append(
        "What about AI's impact on jobs?"
    )

    app.update_state(config, current_state.values)

    # Resume
    await app.ainvoke(None, config)
```

### 3. Sentiment Tracking

```python
def track_sentiment(state: TalkshowState):
    """Track emotional tone of conversation"""
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini")

    recent_messages = state.messages[-5:]

    sentiment_prompt = f"""
    Analyze the sentiment of this conversation:

    {format_messages(recent_messages)}

    For each speaker, rate sentiment (-1 to 1):
    -1 = very negative
    0 = neutral
    1 = very positive

    Also rate:
    - Engagement level (0-1)
    - Tension level (0-1)
    - Agreement level (0-1)

    Respond in JSON format.
    """

    analysis = llm.invoke(sentiment_prompt)
    # Parse and store in state for visualization
    return analysis
```

---

## Production Considerations

### 1. Cost Optimization

```python
# Use cheaper models for routing
def cost_effective_router(state):
    # Use GPT-4o-mini for routing decisions
    llm_router = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

def create_guest(name, expertise):
    # Use GPT-4o for guest responses (quality matters)
    llm_guest = ChatOpenAI(model="gpt-4o", temperature=0.8)
```

### 2. Context Window Management

```python
def summarize_old_messages(state: TalkshowState):
    """Prevent context window overflow"""
    if len(state.messages) > 50:
        # Summarize messages 0-30
        old_messages = state.messages[:30]

        summary_prompt = f"""
        Summarize this talkshow conversation:
        {format_messages(old_messages)}

        Provide:
        - Key points discussed
        - Main arguments from each guest
        - Important questions asked

        Keep summary under 200 words.
        """

        summary = llm.invoke(summary_prompt).content

        # Replace old messages with summary
        return {
            "messages": [
                AIMessage(content=f"[Earlier discussion summary: {summary}]", name="system"),
                *state.messages[30:]  # Keep recent messages
            ]
        }

    return state
```

### 3. Fail-Safe Mechanisms

```python
def safe_router_with_fallback(state: PanelState):
    """Router with automatic fallback"""
    try:
        next_speaker = llm_panel_router(state)
    except Exception as e:
        logger.error(f"Router failed: {e}")
        # Fallback to round-robin
        idx = state.current_turn % len(state.panelists)
        next_speaker = state.panelists[idx]

    return next_speaker

def bounded_conversation(state):
    """Ensure conversation terminates"""
    MAX_TURNS = 30  # Hard limit
    STALL_THRESHOLD = 5  # Detect stuck loops

    if state.current_turn >= MAX_TURNS:
        logger.warning("Max turns reached, ending conversation")
        return END

    # Detect if same speakers repeating
    if len(state.speaking_history) >= STALL_THRESHOLD:
        recent = state.speaking_history[-STALL_THRESHOLD:]
        if len(set(recent)) == 1:  # Same speaker repeated
            logger.warning("Detected stall, inserting moderator")
            return "moderator"

    return normal_routing(state)
```

### 4. Observability

```python
import structlog

logger = structlog.get_logger()

def logged_agent(agent_func):
    """Wrapper to log all agent invocations"""
    def wrapper(state):
        logger.info(
            "agent_invoked",
            agent=agent_func.__name__,
            turn=state.get("current_turn", 0),
            message_count=len(state.get("messages", []))
        )

        start_time = time.time()
        result = agent_func(state)
        duration = time.time() - start_time

        logger.info(
            "agent_completed",
            agent=agent_func.__name__,
            duration=duration
        )

        return result

    return wrapper

# Use decorator
@logged_agent
def host_agent(state):
    # ... agent logic
```

---

## Complete Production Example

Here's a production-ready talkshow with all features:

```python
# See the full implementation in the first section above
# Additional production enhancements:

class ProductionTalkshowState(TalkshowState):
    # Add production fields
    error_count: int = 0
    max_errors: int = 3
    conversation_sentiment: dict = {}
    processing_times: list[float] = []

# Full production graph with monitoring
def build_production_talkshow():
    graph = StateGraph(ProductionTalkshowState)

    # Wrap all agents with error handling and logging
    graph.add_node("host", with_error_handling(with_logging(host_agent)))
    graph.add_node("guest_1", with_error_handling(with_logging(guest_1)))
    # ...

    # Add checkpointing for recovery
    from langgraph.checkpoint.postgres import PostgresSaver
    checkpointer = PostgresSaver.from_conn_string(DB_URL)

    return graph.compile(checkpointer=checkpointer)
```

---

## Summary

**Key Takeaways for Talkshow Implementations**:

1. **State Design**: Track conversation metadata (turns, speakers, topics)
2. **Dynamic Routing**: Use LLM for organic conversation or structured turn-taking
3. **Context Management**: Summarize old messages to avoid overflow
4. **Error Handling**: Fallbacks for routing failures
5. **Monitoring**: Log all agent calls and track performance
6. **Human-in-the-Loop**: Allow audience participation and moderator intervention
7. **Production Ready**: Use checkpointing, error recovery, and observability

**Performance**:
- Use GPT-4o-mini for routing (fast, cheap)
- Use GPT-4o/Claude for agent responses (quality)
- Implement context summarization at ~50 messages
- Set hard turn limits (20-30 turns typical)

**Best Architecture**: Shared scratchpad (collaborative) for debates and panels where all agents benefit from seeing full conversation history.
