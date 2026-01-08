# MARIE - Multi-Agent Research Intelligence Expert
## System Requirements Specification

### Version: 2.0
### Date: January 2026

---

## 1. Executive Summary

MARIE is a **general-purpose AI assistant** with specialized access to **Colombian scientific production data** through the Impactu platform. The system serves dual purposes:

1. **General AI Assistant**: Answer any question across all domains (science, technology, humanities, etc.)
2. **Colombian Research Expert**: Leverage unique database of Colombian scientific output when relevant

### Key Principles

**🇨🇴 Colombian Scientific Production Database**
- MARIE has exclusive access to comprehensive data on Colombian research output
- Includes institutions, researchers, papers, patents, and scientometric metrics from Colombia

**⚖️ Balanced Information Strategy**
- **Not all questions require RAG** - General knowledge uses LLM directly
- **Scientific topics CAN use RAG** - But it's optional, not mandatory
- **Colombian context ENABLES RAG** - When Colombia is mentioned, database becomes valuable resource
- **Hybrid approach preferred** - Combine LLM knowledge with database facts when beneficial

**🤖 Multi-Agent Intelligence**
The orchestrator must understand:
- When database adds value vs. when it's unnecessary
- Balance between speed (direct LLM) and depth (RAG with evidence)
- Colombian context as trigger for potential database consultation
- Scientific queries as opportunities (not obligations) for RAG

---

## 2. Core Capabilities

### 2.1 General Question Answering
The system MUST be able to respond to:
- **General knowledge questions** (science, history, technology, culture)
- **Technical questions** (programming, mathematics, engineering)
- **Social sciences** (sociology, economics, political science)
- **Humanities** (philosophy, literature, arts)
- **Practical advice** (how-to questions, recommendations)

**No RAG required** for these types of questions (unless Colombian context added).

### 2.2 Colombian Research-Specific Queries
The system has **unique access to Colombian scientific production data** and SHOULD use it when:
- **Colombian institutions mentioned** (direct or abbreviated): "Universidad Nacional", "UdeA", "Javeriana", "Los Andes"
- **Colombian researchers** identified (by name or context)
- **Colombian research groups** or departments
- **Colombian geographic context** in research: "Medellín research center", "Bogotá university", etc.
- **Quantitative queries** about Colombian entities: "How many papers...", "Top researchers..."
- **Comparative analysis** involving Colombian entities

**Key: User does NOT need to say "Colombia" or "Colombian"** - The system must recognize Colombian entities automatically.

**RAG strongly recommended** - Leverage unique Colombian database.

**Examples:**
- ✅ "Papers from UdeA" → RAG (UdeA is Colombian)
- ✅ "Research at Universidad Nacional" → RAG (Colombian institution)
- ✅ "Dr. María García publications" → Check DB, if Colombian → RAG
- ❌ "What is machine learning?" → No RAG (no entities)

### 2.3 General Research Queries (Non-Colombian)
The system CAN answer research questions about:
- International institutions, researchers, topics
- Global scientometric questions
- General research methodology

**RAG optional** - Use if entities exist in database, otherwise LLM knowledge.

### 2.4 Hybrid Queries
The system MUST handle questions that combine general knowledge with Colombian research data:
- "Explain quantum computing and show me Colombian researchers in this field"
- "What is the economic impact of AI? Include Colombian research on this topic"
- "Describe climate change and reference Colombian studies"

**Balanced approach** - Combine LLM knowledge with selective Colombian data retrieval.

---

## 3. Intelligent RAG Decision Making

### 3.1 When to Use RAG

The orchestrator agent MUST activate RAG retrieval when:

✅ **Colombian entities detected (explicit or implicit):**
- Colombian institutions: "Universidad de Antioquia", "UdeA", "Universidad Nacional de Colombia", "UniAndes", "Javeriana", etc.
- Colombian researchers: "Juan Pérez", "María García" (if in Colombian context)
- Colombian research groups or departments
- Colombian geographic locations in research context: "Medellín", "Bogotá", "Cali" + research terms
- **Note:** User does NOT need to say "Colombia" or "Colombian" explicitly - just mentioning the entity is enough

✅ **Query mentions specific entities (any):**
- Any institution name (check if exists in database)
- Any author name (check if exists in database)
- Research groups or departments
- Specific papers, patents, or datasets

✅ **Query requests quantitative metrics:**
- "How many papers...?"
- "Top X most cited..."
- "Compare productivity between..."
- "Citation impact of..."
- "H-index", "Impact factor"

✅ **Query asks for specific research artifacts:**
- "Papers about [topic]" + entity mention
- "Patents in biotechnology [from specific place]"
- "Research on [topic]" + specific context
- "Software projects from..."

✅ **Query requires evidence-based answers:**
- "Show me proof that..."
- "What's the evidence for...?"
- "Reference studies on..."

**🎯 Colombian Context Detection:**
```
Query: "Papers from UdeA" → Detect: UdeA = Colombian → Use RAG
Query: "Research at Universidad Nacional" → Detect: Colombian institution → Use RAG
Query: "Dr. Juan Pérez publications" → Check database → If Colombian, use RAG
Query: "AI research in Medellín" → Detect: Medellín + research → Use RAG
Query: "What is AI?" → No entities detected → Direct LLM (no RAG)
```

**🎯 Balanced Approach:**
- Scientific topics alone: **Direct LLM** (e.g., "What is photosynthesis?")
- Scientific topics + Colombian entity: **Consider RAG** (e.g., "Photosynthesis research at UdeA")
- Colombian entity + quantitative: **Definitely RAG** (e.g., "Top papers from Universidad Nacional")

### 3.2 When NOT to Use RAG

The system MUST respond directly (LLM only) when:

❌ **General knowledge questions:**
- "What is photosynthesis?"
- "Explain machine learning"
- "How does a blockchain work?"

❌ **Opinion or advice questions:**
- "What's the best programming language?"
- "How should I structure my thesis?"
- "Give me recommendations for..."

❌ **Conceptual explanations:**
- "Explain the difference between X and Y"
- "What are the advantages of...?"
- "Why is X important?"

❌ **Greetings and conversational:**
- "Hello, how can you help me?"
- "What can you do?"
- "Thank you"

❌ **Hypothetical or future scenarios:**
- "What would happen if...?"
- "How could we improve...?"
- "What's the future of...?"

### 3.3 Decision Flow

```
Query → Intent Detection (LLM) → Classification
                                      ↓
            ┌─────────────────────────┴──────────────────────────────┐
            ↓                                                          ↓
    Mentions Colombia/Colombian entities?                     General topic?
            ↓                                                          ↓
    YES → Consider RAG                                        NO → Direct LLM
    (Check if adds value)                                     (Fast response)
            ↓
    Needs specific data?
    ├─ YES → RAG Pipeline
    │   - Entity Resolution
    │   - Retrieval from Colombian DB
    │   - Metrics from Impactu
    │   - Evidence-based answer
    │
    └─ NO → Hybrid Response
        - LLM explanation
        - Optional: Add Colombian data if relevant
        - Balanced information
```

**Examples of Balanced Decisions:**

| Query | Decision | Reasoning |
|-------|----------|-----------|
| "What is machine learning?" | Direct LLM | General definition, no Colombian context |
| "Machine learning research in Colombia" | RAG | Colombian context, use database |
| "Explain photosynthesis" | Direct LLM | Biology concept, no research query |
| "Photosynthesis papers from UdeA" | RAG | Specific institution + papers request |
| "Best universities in Colombia" | Hybrid | LLM knowledge + RAG data for rankings |
| "Top Colombian researchers in AI" | RAG | Colombian context + quantitative request |
| "History of Colombia" | Direct LLM | General history, not research-focused |
| "Research on Colombian history" | Consider RAG | Might have papers in database |

---

## 4. Response Format Requirements

### 4.1 Research-Based Responses (with RAG)

MUST include:
- **Answer** with natural language explanation
- **References to Impactu** with URLs when data comes from platform
- **Confidence score** with reasoning
- **Evidence items** from database
- **Citation counts** when relevant (but not the focus)
- **Impactu links** format: `https://impactu.colav.co/[entity_type]/[entity_id]`

Example:
```
## Analysis of AI Research at Universidad de Antioquia

The institution has produced 45 papers on Artificial Intelligence in the last 5 years...

**Key Researchers:**
- Dr. John Doe (15 papers, h-index: 8) - [View profile](https://impactu.colav.co/person/123)
- Dr. Jane Smith (12 papers) - [View profile](https://impactu.colav.co/person/456)

**Top Papers:**
1. "Neural Networks for X" (2023) - 23 citations
   [View in Impactu](https://impactu.colav.co/work/789)

🟢 Confidence: 92% - Based on 45 research artifacts from Impactu database
```

### 4.2 General Knowledge Responses (no RAG)

MUST include:
- **Direct answer** based on LLM knowledge
- **Clear explanation** without unnecessary jargon
- **Examples** when helpful
- **NO confidence score** (not needed)
- **NO citations** (unless referring to well-known facts)
- **NO Impactu links** (no database used)

Example:
```
## What is Machine Learning?

Machine learning is a subset of artificial intelligence that enables computers 
to learn from data without being explicitly programmed. Instead of following 
fixed rules, ML algorithms identify patterns and make decisions based on examples.

**Key Concepts:**
- Supervised Learning: Learning from labeled examples
- Unsupervised Learning: Finding patterns in unlabeled data
- Reinforcement Learning: Learning through trial and error

**Common Applications:**
- Image recognition
- Natural language processing
- Recommendation systems
- Fraud detection
```

### 4.3 Hybrid Responses

MUST combine both approaches:
- Start with LLM explanation (general knowledge)
- Follow with research data when relevant
- Include Impactu references only for data-backed claims

---

## 5. Multi-Agent Architecture (Magentic)

MARIE implements the **Magentic Architecture** for multi-agent orchestration with human-in-the-loop capabilities.

### 5.1 Agent Roles

#### Orchestrator Agent (Always Active)
- **Responsibility:** Query understanding, intent detection, **Colombian entity recognition**, routing decisions
- **LLM Usage:** Parse query, detect entities (especially Colombian), determine if RAG is needed
- **Colombian Detection:** Must recognize Colombian institutions, researchers, locations without explicit "Colombia" mention
- **Output:** Routing decision (RAG pipeline vs direct response)
- **Magentic Role:** Central coordinator, plan creator, human collaboration manager

#### Entity Resolution Agent (Conditional)
- **Activation:** Only when specific entities mentioned (especially Colombian)
- **Responsibility:** Disambiguate institutions, authors, groups
- **Data Source:** MongoDB (institutions, persons collections)
- **Colombian Focus:** Prioritize Colombian entity resolution
- **Magentic Role:** Specialized agent with ledger state updates

#### Retrieval Agent (Conditional)
- **Activation:** Only when research data needed from database
- **Responsibility:** Fetch papers, datasets, patents (primarily Colombian)
- **Data Source:** OpenSearch (works) + MongoDB (metadata)
- **Magentic Role:** RAG specialist agent with evidence collection

#### Metrics Agent (Conditional)
- **Activation:** Only when quantitative analysis requested
- **Responsibility:** Compute h-index, productivity, impact metrics
- **Data Source:** MongoDB aggregations
- **Magentic Role:** Analytics specialist with ledger tracking

#### Citations Agent (Conditional)
- **Activation:** Only when evidence mapping needed for research responses
- **Responsibility:** Map evidence to claims, build reference list with Impactu URLs
- **Data Source:** Retrieved documents
- **Magentic Role:** Evidence tracking and citation management

#### Reporting Agent (Always Active)
- **Responsibility:** Format final response based on query type
- **Modes:** 
  - **General mode:** Clean explanation, no citations (for general knowledge)
  - **Research mode:** Evidence-based with citations and Impactu links (for Colombian research)
  - **Hybrid mode:** Combination of explanation + selective references
- **Magentic Role:** Final synthesis with confidence assessment

### 5.2 Routing Logic (Magentic Decision Flow)

```python
def route_query(parsed_query: Dict) -> List[str]:
    """
    Magentic orchestrator routing with Colombian entity awareness.
    
    Returns:
        List of agent names to execute
    """
    agents = ["orchestrator"]  # Always runs (Magentic coordinator)
    
    # Check if Colombian entities or research data needed
    if has_colombian_entities(parsed_query) or needs_research_data(parsed_query):
        # Activate RAG pipeline
        agents.extend([
            "entity_resolution",  # Colombian entity disambiguation
            "retrieval",          # Database access
            "metrics",            # Quantitative analysis
            "citations"           # Evidence mapping
        ])
    
    # Always format final response
    agents.append("reporting")
    
    return agents

def has_colombian_entities(parsed_query: Dict) -> bool:
    """
    Detect Colombian entities without explicit "Colombia" mention.
    """
    entities = parsed_query.get("entities", {})
    
    # Check for Colombian institutions
    colombian_institutions = [
        "Universidad de Antioquia", "UdeA", "Universidad Nacional",
        "Javeriana", "UniAndes", "Universidad de los Andes",
        "Universidad del Valle", "EAFIT", "UPB", # etc.
    ]
    
    # Check query text for Colombian markers
    query_text = parsed_query.get("original_query", "").lower()
    
    for institution in colombian_institutions:
        if institution.lower() in query_text:
            return True
    
    # Check parsed entities
    if entities.get("institutions") or entities.get("authors"):
        # Cross-reference with Colombian database
        return True  # Let entity_resolution agent determine
    
    return False

def needs_research_data(parsed_query: Dict) -> bool:
    """
    Determine if query requires database access (beyond Colombian check).
    """
    # Quantitative requests
    if parsed_query.get("intent") in ["metrics", "comparison", "ranking"]:
        return True
    
    # Specific artifact requests
    if parsed_query.get("artifact_type") in ["paper", "patent", "dataset"]:
        return True
    
    # General knowledge - no RAG
    if parsed_query.get("intent") in ["explanation", "definition", "greeting"]:
        return False
    
    # Default: check for research indicators
    return False
```

### 5.3 Magentic Architecture Compliance

**Core Orchestration:** ✅
- Central orchestrator with control flow
- LLM-powered plan creation with Colombian awareness
- Conditional routing based on entity detection
- Task delegation to specialized agents

**Ledger (State Management):** ✅  
- Shared AgentState across all agents
- Task tracking lifecycle (create, start, complete, fail)
- Evidence map with Colombian source traceability
- Audit trail for all decisions

**Human-in-the-Loop (Magentic-UI):** ✅
- Co-Planning: Collaborate on complex Colombian research queries
- Co-Tasking: Human completes ambiguous entity resolution
- Action Guards: Approve critical database operations
- Verification: Validate results with low confidence

**Evidence & Citations:** ✅
- Evidence collection from Colombian database
- Citation mapping with Impactu URLs
- Source traceability (MongoDB/OpenSearch)
- Confidence assessment with LLM reasoning

**Advanced Features:** ✅
- Multi-turn conversations with session memory
- Long-term memory persistence
- Context preservation for Colombian research context
- Learning from feedback on entity recognition

---

## 6. User Personas

### 6.1 Researcher / Scientist
- **Needs:** Scientometric analysis, collaboration opportunities, impact assessment
- **Queries:** Research-heavy, requires RAG
- **Response:** Full citations, metrics, Impactu links

### 6.2 University Administrator
- **Needs:** Institutional reports, rankings, productivity metrics
- **Queries:** Quantitative, comparative analysis
- **Response:** Data-driven, visual-ready, exportable

### 6.3 Student
- **Needs:** Learn about topics, find relevant papers, understand concepts
- **Queries:** Mix of general + research
- **Response:** Educational tone, examples, selective references

### 6.4 General User
- **Needs:** Quick answers, explanations, recommendations
- **Queries:** Mostly general knowledge
- **Response:** Direct, concise, no unnecessary citations

### 6.5 Policy Maker / Decision Maker
- **Needs:** Evidence-based insights, trends, impact analysis
- **Queries:** Hybrid (policy + research evidence)
- **Response:** Executive summary + supporting research data

---

## 7. Non-Functional Requirements

### 7.1 Performance
- **General queries:** < 2 seconds response time
- **Research queries:** < 10 seconds response time
- **Complex analysis:** < 30 seconds response time

### 7.2 Accuracy
- **General knowledge:** 95%+ accuracy (LLM dependent)
- **Research data:** 100% accuracy (database verified)
- **Entity resolution:** 90%+ precision (with human-in-loop fallback)

### 7.3 Transparency
- **Always indicate** data source (LLM knowledge vs Impactu database)
- **Confidence scores** for research-backed claims
- **No citations** when not needed (avoid overwhelming user)

### 7.4 Adaptability
- System must **learn from feedback** which queries need RAG
- **Improve routing decisions** over time
- **Track false positives** (unnecessary RAG activations)

---

## 8. Example Scenarios

### Scenario 1: Pure General Knowledge
**Query:** "What is the difference between supervised and unsupervised learning?"

**Expected Behavior:**
- ❌ No RAG activation
- ✅ Direct LLM response
- ✅ Clear explanation with examples
- ❌ No citations
- ❌ No Impactu links

### Scenario 2: Pure Research Query
**Query:** "Top 10 most cited papers from Universidad Nacional de Colombia in 2023"

**Expected Behavior:**
- ✅ Full RAG pipeline
- ✅ Entity resolution (universidad)
- ✅ Retrieval (papers 2023)
- ✅ Metrics (citation counts)
- ✅ Citations with Impactu links
- ✅ Confidence score

### Scenario 3: Hybrid Query
**Query:** "Explain machine learning and show me research from Latin America on this topic"

**Expected Behavior:**
- ✅ Part 1: Direct LLM explanation (no RAG)
- ✅ Part 2: RAG pipeline for Latin American research
- ✅ Combined response with clear sections
- ✅ Citations only for research part

### Scenario 4: Greeting
**Query:** "Hello! How can you help me?"

**Expected Behavior:**
- ❌ No RAG
- ✅ Welcome message
- ✅ Explain capabilities (both general + research)
- ❌ No citations

---

## 9. Success Metrics

### 9.1 User Satisfaction
- **Response relevance:** > 90%
- **Response completeness:** > 85%
- **User ratings:** > 4.5/5

### 9.2 System Efficiency
- **Unnecessary RAG calls:** < 10%
- **Missed RAG opportunities:** < 5%
- **Average response time:** < 5 seconds

### 9.3 Data Quality
- **Citation accuracy:** 100%
- **Impactu link validity:** 100%
- **Entity resolution precision:** > 90%

---

## 10. Future Enhancements

### 10.1 Multi-modal Responses
- Generate charts and visualizations for metrics
- Create network graphs for collaborations
- Export reports in PDF/Excel

### 10.2 Proactive Insights
- Alert users to new relevant papers
- Suggest collaboration opportunities
- Identify emerging trends

### 10.3 Customization
- User preferences for detail level
- Domain-specific response templates
- Citation style preferences (APA, IEEE, etc.)

---

## 11. Implementation Notes

### Critical: Intent Classification
The **most important** part of the system is accurate intent detection:
- Use LLM to classify query intent
- Train on examples of research vs general queries
- Continuously improve classification based on feedback

### Impactu Integration
- Only reference Impactu when data actually comes from database
- Use proper URL format: `https://impactu.colav.co/{entity_type}/{id}`
- Entity types: `person`, `work`, `affiliation`, `group`

### Citation Philosophy
- **Not all answers need citations**
- Only cite when claiming specific research findings
- General knowledge doesn't need academic references
- User experience > academic rigor (for general queries)

---

## 12. System Boundaries

### In Scope
✅ Answer any question (general or research)
✅ Intelligent RAG activation
✅ Multi-domain knowledge
✅ Evidence-based claims when needed
✅ Conversational interaction

### Out of Scope
❌ Real-time web search
❌ Document upload/analysis
❌ Writing full research papers
❌ Peer review or quality assessment
❌ Financial or medical advice

---

**Document Status:** Living document - to be updated as system evolves
**Next Review:** After user feedback and usage analytics
