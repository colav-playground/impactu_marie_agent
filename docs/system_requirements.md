# MARIE - Multi-Agent Research Intelligence Expert
## System Requirements Specification

### Version: 2.0
### Date: January 2026

---

## 1. Executive Summary

MARIE is a general-purpose AI assistant with specialized capabilities in research information and scientometric analysis. The system must be capable of answering **any type of question** - from general knowledge to highly specialized queries across multiple domains - while intelligently deciding when to leverage research data from Impactu platform.

### Key Principle
**Not all questions require RAG (Retrieval Augmented Generation).** The system must autonomously determine when to use research data versus when to provide direct responses based on the LLM's knowledge.

---

## 2. Core Capabilities

### 2.1 General Question Answering
The system MUST be able to respond to:
- **General knowledge questions** (science, history, technology, culture)
- **Technical questions** (programming, mathematics, engineering)
- **Social sciences** (sociology, economics, political science)
- **Humanities** (philosophy, literature, arts)
- **Practical advice** (how-to questions, recommendations)

**No RAG required** for these types of questions.

### 2.2 Research-Specific Queries
The system MUST be able to answer:
- Scientometric analysis (citations, h-index, impact factors)
- Author productivity and collaboration patterns
- Institutional research output
- Research trends and emerging topics
- Funding and grant analysis

**RAG required** - Must retrieve data from Impactu MongoDB and OpenSearch.

### 2.3 Hybrid Queries
The system MUST handle questions that combine general knowledge with research data:
- "Explain quantum computing and show me top researchers in this field"
- "What is the economic impact of AI? Include research on this topic"
- "Describe climate change policies and reference relevant studies"

**Partial RAG** - Combine LLM knowledge with selective data retrieval.

---

## 3. Intelligent RAG Decision Making

### 3.1 When to Use RAG

The orchestrator agent MUST activate RAG retrieval when:

✅ **Query mentions specific entities:**
- Institution names: "Universidad de Antioquia", "MIT", "Oxford"
- Author names: "Dr. John Smith", "María García"
- Research groups or departments
- Specific papers, patents, or datasets

✅ **Query requests quantitative metrics:**
- "How many papers...?"
- "Top X most cited..."
- "Compare productivity between..."
- "Citation impact of..."
- "H-index", "Impact factor"

✅ **Query asks for specific research artifacts:**
- "Papers about quantum computing"
- "Patents in biotechnology"
- "Datasets on climate change"
- "Software projects from..."

✅ **Query requires evidence-based answers:**
- "Show me proof that..."
- "What's the evidence for...?"
- "Reference studies on..."

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
            ┌─────────────────────────┴──────────────────────┐
            ↓                                                  ↓
    Needs Research Data?                               General Knowledge?
            ↓                                                  ↓
    YES → RAG Pipeline                                 NO → Direct LLM Response
    - Entity Resolution                                - No database access
    - Retrieval                                        - Fast response
    - Metrics                                          - No citations needed
    - Citations                                        
    - Evidence-based answer                            
```

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

## 5. Multi-Agent Architecture

### 5.1 Agent Roles

#### Orchestrator Agent (Always Active)
- **Responsibility:** Query understanding, intent detection, routing decisions
- **LLM Usage:** Parse query, determine if RAG is needed
- **Output:** Routing decision (RAG pipeline vs direct response)

#### Entity Resolution Agent (Conditional)
- **Activation:** Only when specific entities mentioned
- **Responsibility:** Disambiguate institutions, authors, groups
- **Data Source:** MongoDB (institutions, persons collections)

#### Retrieval Agent (Conditional)
- **Activation:** Only when research data needed
- **Responsibility:** Fetch papers, datasets, patents
- **Data Source:** OpenSearch (works) + MongoDB (metadata)

#### Metrics Agent (Conditional)
- **Activation:** Only when quantitative analysis requested
- **Responsibility:** Compute h-index, productivity, impact metrics
- **Data Source:** MongoDB aggregations

#### Citations Agent (Conditional)
- **Activation:** Only when evidence mapping needed
- **Responsibility:** Map evidence to claims, build reference list
- **Data Source:** Retrieved documents

#### Reporting Agent (Always Active)
- **Responsibility:** Format final response based on query type
- **Modes:** 
  - Research mode (with citations, Impactu links)
  - General mode (clean explanation, no refs)
  - Hybrid mode (combination)

### 5.2 Routing Logic

```python
def route_query(parsed_query: Dict) -> List[str]:
    """
    Determine which agents to activate based on query analysis.
    
    Returns:
        List of agent names to execute
    """
    agents = ["orchestrator"]  # Always runs
    
    # Check if research data needed
    if needs_research_data(parsed_query):
        agents.extend([
            "entity_resolution",
            "retrieval",
            "metrics",
            "citations"
        ])
    
    # Always format final response
    agents.append("reporting")
    
    return agents

def needs_research_data(parsed_query: Dict) -> bool:
    """
    Determine if query requires database access.
    """
    # Check for entity mentions
    if parsed_query.get("entities"):
        return True
    
    # Check for quantitative requests
    if parsed_query.get("intent") in ["metrics", "comparison", "ranking"]:
        return True
    
    # Check for specific artifact requests
    if parsed_query.get("artifact_type") in ["paper", "patent", "dataset"]:
        return True
    
    # Default to general knowledge
    return False
```

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
