# core/extractor.py — AI extraction functions powered by Mistral via LangChain

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os


def get_llm(temperature: float = 0.2):
    model = os.getenv("MISTRAL_MODEL", "open-mistral-7b")
    return ChatMistralAI(
        model=model,
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=temperature,
    )


def build_chain(system_prompt: str, temperature: float = 0.2):
    llm = get_llm(temperature)
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )


# ─── Original Extractions ────────────────────────────────────────────────────────

def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )
    return chain.invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    return chain.invoke(transcript)


def extract_topics(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "identify the top 5-7 main topics or themes discussed. "
        "For each topic provide a brief 1-line description. "
        "Format each as: '🏷️ Topic Name: brief description' on its own line. "
        "If unclear, return 'No distinct topics identified.'"
    )
    return chain.invoke(transcript)


# ─── New Extractions ─────────────────────────────────────────────────────────────

def generate_tldr(transcript: str) -> str:
    """Generate a 3-sentence TL;DR executive brief."""
    chain = build_chain(
        "You are a sharp executive assistant. Read the meeting transcript and "
        "write exactly 3 crisp sentences that capture: (1) what the meeting was about, "
        "(2) the single most important decision or outcome, and "
        "(3) the most critical next step. "
        "No bullet points — plain flowing sentences only. Be concise and specific.",
        temperature=0.3,
    )
    return chain.invoke(transcript[:4000])


def extract_named_entities(transcript: str) -> str:
    """Extract people, organizations, tools/technologies, and dates."""
    chain = build_chain(
        "You are an information extraction expert. From the meeting transcript extract:\n"
        "👤 PEOPLE: Names of all individuals mentioned (with role if stated)\n"
        "🏢 ORGANIZATIONS: Companies, teams, or departments mentioned\n"
        "🛠️ TOOLS & TECH: Software, platforms, systems, or technologies mentioned\n"
        "📅 DATES & DEADLINES: Any specific dates, timelines, or deadlines mentioned\n\n"
        "Format each category as a comma-separated list. "
        "If a category has no entries, write 'None mentioned'. "
        "Keep it factual — only include what is explicitly stated."
    )
    return chain.invoke(transcript)


def extract_risks(transcript: str) -> str:
    """Extract risks, blockers, and concerns with severity assessment."""
    chain = build_chain(
        "You are a risk analyst reviewing a meeting transcript. Extract all risks, "
        "blockers, concerns, or issues mentioned. For each item:\n"
        "- Risk/Issue: what the risk is\n"
        "- Severity: HIGH / MEDIUM / LOW\n"
        "- Mitigation mentioned: (if discussed, else 'Not discussed')\n\n"
        "Format as a numbered list. "
        "If no risks found, say 'No risks or blockers identified.'"
    )
    return chain.invoke(transcript)


def generate_followup_email(title: str, summary: str, action_items: str, decisions: str) -> str:
    """Generate a professional follow-up email from meeting data."""
    llm = get_llm(temperature=0.4)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a professional executive assistant. Write a polished, concise "
         "follow-up email based on the meeting information provided. "
         "The email should: greet the recipients, briefly state the meeting purpose, "
         "list key decisions made, list action items with owners, and close with a "
         "professional sign-off. Keep the tone professional but warm. "
         "Use clear formatting with short sections. Do NOT add placeholder names — "
         "use 'Team' for recipients and 'Best regards, [Your Name]' at the end."),
        ("human",
         "Meeting Title: {title}\n\n"
         "Summary:\n{summary}\n\n"
         "Key Decisions:\n{decisions}\n\n"
         "Action Items:\n{action_items}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "title": title,
        "summary": summary[:1500],
        "decisions": decisions[:1000],
        "action_items": action_items[:1000],
    })


def generate_formal_minutes(title: str, transcript: str, decisions: str, action_items: str, timestamp: str) -> str:
    """Generate formal meeting minutes document."""
    llm = get_llm(temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a professional meeting secretary. Generate formal meeting minutes "
         "in a structured format including: Meeting title, Date, Attendees (extract from transcript), "
         "Agenda items discussed, Discussion points, Decisions made, Action items with owners, "
         "and Next meeting (if mentioned). Use formal language and clear structure. "
         "Format with clear headings and numbered sections."),
        ("human",
         "Meeting Title: {title}\n"
         "Date: {timestamp}\n\n"
         "Transcript excerpt:\n{transcript}\n\n"
         "Decisions:\n{decisions}\n\n"
         "Action Items:\n{action_items}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "title": title,
        "timestamp": timestamp,
        "transcript": transcript[:3000],
        "decisions": decisions[:800],
        "action_items": action_items[:800],
    })


def translate_text(text: str, target_language: str) -> str:
    """Translate text to the target language."""
    chain = build_chain(
        f"You are a professional translator. Translate the following text to {target_language}. "
        "Preserve the meaning, tone, and structure. Only return the translated text — no explanations.",
        temperature=0.1,
    )
    return chain.invoke(text[:3000])


def custom_extract(transcript: str, user_prompt: str) -> str:
    """Run a user-defined extraction prompt over the transcript."""
    llm = get_llm(temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert meeting analyst. Answer the user's extraction request "
         "strictly based on the meeting transcript provided. Be accurate, structured, "
         "and cite specific content from the transcript where possible."),
        ("human", "Transcript:\n{transcript}\n\nRequest: {user_prompt}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"transcript": transcript[:4000], "user_prompt": user_prompt})


def generate_chapters(transcript: str, duration_min: int = 15) -> str:
    """Generates logical video/meeting chapters with estimated timestamps."""
    chain = build_chain(
        f"You are a professional video editor and meeting organizer. Analyze the transcript "
        f"and generate 4 to 8 logical, chronological chapters spanning an estimated ~{duration_min} minute session. "
        f"For each chapter, provide:\n"
        f"- [mm:ss] Chapter Title: 1-sentence summary of what is discussed.\n"
        f"Start with [00:00] for the first chapter. Ensure timestamps are plausible and spaced across the session. "
        f"Format as a list. Only return the chapters."
    )
    return chain.invoke(transcript)


def generate_presentation_outline(title: str, summary: str, decisions: str, action_items: str) -> str:
    """Generates a structured 5-slide executive presentation deck outline."""
    llm = get_llm(temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a management consultant. Convert the meeting summary and decisions into a "
         "compelling 5-slide executive presentation deck outline.\n"
         "Structure:\n"
         "Slide 1: Executive Title & Context\n"
         "Slide 2: Background & Current Challenges\n"
         "Slide 3: Core Discussion & Strategic Findings\n"
         "Slide 4: Key Decisions Made & Rationale\n"
         "Slide 5: Action Plan, Owners & Next Milestones\n\n"
         "For each slide include: Title, 3-4 bullet points, and a 'Speaker Note' tip. Be professional and concise."),
        ("human",
         f"Meeting: {title}\nSummary: {summary[:1500]}\nDecisions: {decisions[:800]}\nActions: {action_items[:800]}")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({})


def generate_jira_tickets(action_items: str) -> str:
    """Converts meeting action items into ready-to-paste Jira/Linear engineering/business tickets."""
    llm = get_llm(temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a technical project manager. Transform the provided meeting action items into "
         "clean, standardized Jira/Linear tickets. For each task output:\n"
         "### [TICKET] <Concise Action Title>\n"
         "- **Assignee:** <Name or 'Unassigned'>\n"
         "- **Priority:** <High / Medium / Low>\n"
         "- **Description:** <Context and task details>\n"
         "- **Acceptance Criteria:**\n"
         "  - [ ] <Criteria 1>\n"
         "  - [ ] <Criteria 2>\n\n"
         "Ensure each item has clear acceptance criteria. Return markdown."),
        ("human", "Action Items:\n{action_items}")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"action_items": action_items[:2000]})


def generate_meeting_scorecard(transcript: str, word_count: int, duration_min: int) -> str:
    """Evaluates meeting effectiveness, actionability, and provides productivity scores."""
    chain = build_chain(
        f"You are an executive productivity coach. Analyze this meeting transcript (Total words: {word_count:,}, ~{duration_min} minutes) "
        "and produce an objective Meeting Effectiveness Scorecard in Markdown:\n\n"
        "### 🎯 Meeting Effectiveness Scorecard\n"
        "- **Overall Productivity Score:** [X/100]\n"
        "- **Actionability Rating:** [High / Medium / Low] (Ratio of concrete next steps vs general discussion)\n"
        "- **Clarity & Focus Score:** [X/100] (Did the discussion stay on topic?)\n"
        "- **Pacing & Discussion Density:** [Balanced / Rushed / Wandering]\n\n"
        "### 💡 Top 3 Recommendations for Next Meeting\n"
        "1. <Specific tip to save time or improve decision-making>\n"
        "2. <Specific tip regarding follow-ups or structure>\n"
        "3. <Specific tip on engagement or agenda control>\n\n"
        "Be constructive, analytical, and professional.",
        temperature=0.3
    )
    return chain.invoke(transcript[:4000])


def compare_meetings(session_a: dict, session_b: dict) -> str:
    """Compares two meeting sessions to highlight progress, recurring themes, and sentiment shift."""
    llm = get_llm(temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an executive meeting auditor. Compare the two provided meetings and generate a comparative summary in Markdown:\n\n"
         "### 🔄 Comparative Executive Analysis\n"
         "- **Core Focus Shift:** How did the discussion topics shift between Meeting 1 and Meeting 2?\n"
         "- **Decisions & Momentum:** What new decisions were reached in Meeting 2 vs Meeting 1?\n"
         "- **Action Item Continuity:** Were tasks from Meeting 1 followed up or referenced in Meeting 2?\n"
         "- **Sentiment & Dynamics:** Compare the tone, urgency, or alignment between both sessions.\n\n"
         "Be concise, insightful, and formatted with clear bullet points."),
        ("human",
         f"MEETING 1: {session_a.get('title', 'Meeting 1')} ({session_a.get('timestamp', 'N/A')})\n"
         f"Summary 1:\n{session_a.get('summary', '')[:1000]}\n"
         f"Decisions 1:\n{session_a.get('key_decisions', '')[:500]}\n\n"
         f"MEETING 2: {session_b.get('title', 'Meeting 2')} ({session_b.get('timestamp', 'N/A')})\n"
         f"Summary 2:\n{session_b.get('summary', '')[:1000]}\n"
         f"Decisions 2:\n{session_b.get('key_decisions', '')[:500]}")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({})
