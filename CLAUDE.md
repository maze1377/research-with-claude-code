# Project Guidelines

**Purpose:** Contribution guidelines for the Agentic AI Developer Onboarding Guide

---

## Document Standards

### File Structure
- Use Markdown with clear hierarchical headers (H1 → H2 → H3)
- Include "Last Updated" date at the top of each document
- Add Table of Contents for documents exceeding 300 lines
- Use code blocks with language specification (```python, ```typescript)

### Formatting
- Tables for comparisons and quick references
- Code examples with comments explaining key concepts
- ASCII diagrams for architecture visualization
- Bullet points for lists, numbered for sequential steps

### Cross-Referencing
- Link to related documents using relative paths
- Add "Related Documents" section at document end
- Use anchor links for specific sections

---

## Content Principles

### 1. Practical Over Theoretical
- Lead with actionable information
- Include working code examples where applicable
- Provide copy-paste ready snippets

### 2. Current and Accurate
- Use December 2025 information and pricing
- Update documents when frameworks release major versions
- Reference authoritative sources (Anthropic, OpenAI, LangChain docs)

### 3. Tiered for All Levels
- Mark content as Beginner, Intermediate, or Advanced
- Provide "skip to" links for experienced readers
- Include prerequisites at section start

### 4. Production-Focused
- Emphasize patterns that work at scale
- Include failure cases and how to avoid them
- Provide cost and performance considerations

---

## File Organization

### Directory Structure
```
/
├── phase-0-prerequisites/      # Entry point for beginners
├── phase-1-foundations/        # Core concepts
├── phase-2-building-agents/    # First agents
├── phase-3-patterns/           # Multi-agent, workflows
├── phase-4-production/         # Testing, deployment
├── phase-5-security-compliance/# Security, governance
├── phase-6-advanced/           # Mastery topics
├── product-strategy/           # Business guidance
├── developer-productivity/     # Tools and workflows
└── reference/                  # Quick lookup
```

### File Naming
- Use lowercase with hyphens: `tool-development-guide.md`
- Be descriptive: `memory-systems-guide.md` not `memory.md`
- Keep names under 30 characters when possible

---

## Content Updates

### When to Update
- New framework major versions (e.g., LangGraph 2.0)
- New model releases affecting recommendations
- New security vulnerabilities or best practices
- Community feedback on unclear sections

### Update Process
1. Check if content already exists in another document
2. Update the document's "Last Updated" date
3. Update cross-references if structure changes
4. Run link validator before committing

---

## Code Examples

### Requirements
- Must be syntactically correct
- Include necessary imports
- Add error handling for production examples
- Specify framework/library versions

### Format
```python
# Good: Includes imports, typing, error handling
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="Search query")

@tool(args_schema=SearchInput)
def search(query: str) -> str:
    """Search for information."""
    try:
        return perform_search(query)
    except Exception as e:
        return f"Search failed: {e}"
```

---

## Quality Checklist

Before submitting changes:
- [ ] All code examples tested
- [ ] All internal links work
- [ ] Consistent formatting with existing docs
- [ ] "Last Updated" date updated
- [ ] Related Documents section updated
- [ ] No time estimates (phases only)

---

## Style Guide

### Terminology
- Use "agent" not "AI agent" (redundant in this context)
- Use "tool" not "function" for agent capabilities
- Use "LLM" on first use, can abbreviate after
- Use framework names as proper nouns (LangGraph, CrewAI)

### Voice
- Second person for instructions ("You should...")
- Present tense for descriptions
- Active voice over passive
- Concise sentences

### Avoid
- Marketing language ("revolutionary", "game-changing")
- Time estimates ("takes 2 weeks")
- Absolute claims ("the best", "always")
- Outdated information (check dates)

---

## Related Documents

- [README.md](README.md) - Project overview and navigation
- [COMPETENCY-CHECKLIST.md](COMPETENCY-CHECKLIST.md) - Skills tracking
- [task.md](task.md) - Research tracking and status
