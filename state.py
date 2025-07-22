"""
AutoGen Deep Research State Management Module

This module defines all data models and message protocols for the Deep Research system
using Microsoft AutoGen framework. It implements structured communication between agents
using Pydantic models and dataclasses.
"""

from typing import List, Optional, Dict, Any, Annotated
# from dataclasses import dataclass, field  # Not needed with Pydantic BaseModel
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import operator


# =====================================================
# Message Protocol Definitions (AutoGen Pattern)
# =====================================================
# These messages define the communication protocol between agents
# following AutoGen's message-passing architecture

class UserRequest(BaseModel):
    """Initial user request for research"""
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    context: List[Dict[str, Any]] = Field(default_factory=list)
    

class ClarificationRequest(BaseModel):
    """Request for user clarification"""
    need_clarification: bool
    question: str
    verification: str
    original_request: str
    

class ClarificationResponse(BaseModel):
    """User's response to clarification request"""
    content: str
    request_id: str
    

class ResearchBrief(BaseModel):
    """Processed research brief from user request"""
    research_brief: str
    original_request: str
    clarifications: List[str] = Field(default_factory=list)
    

class ResearchTask(BaseModel):
    """Task assignment for individual researcher"""
    topic: str
    subtopics: List[str] = Field(default_factory=list)
    context: List[Dict[str, Any]] = Field(default_factory=list)
    max_iterations: int = 5
    assigned_to: Optional[str] = None
    

class ResearchResult(BaseModel):
    """Result from individual researcher"""
    topic: str
    findings: str
    sources: List[str] = Field(default_factory=list)
    raw_notes: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    researcher_id: str
    iterations_used: int
    

class CompressedResearch(BaseModel):
    """Compressed research output"""
    summary: str
    key_findings: List[str]
    methodology: str
    combined_sources: List[str]
    topics_covered: List[str]
    

class FinalReport(BaseModel):
    """Final research report"""
    title: str
    executive_summary: str
    detailed_findings: str
    conclusion: str
    recommendations: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =====================================================
# Workflow State Definitions
# =====================================================
# These define the state management for the research workflow

class WorkflowStage(Enum):
    """Stages of the research workflow"""
    INITIALIZATION = "initialization"
    CLARIFICATION = "clarification"
    PLANNING = "planning"
    RESEARCH = "research"
    COMPRESSION = "compression"
    REPORT_GENERATION = "report_generation"
    COMPLETED = "completed"
    

class ResearchState(BaseModel):
    """
    Main state container for the research workflow.
    This replaces LangGraph's MessagesState pattern with AutoGen's approach.
    """
    # Workflow metadata
    workflow_id: str
    stage: WorkflowStage = WorkflowStage.INITIALIZATION
    
    # User interaction
    user_request: Optional[UserRequest] = None
    clarification_requests: List[ClarificationRequest] = Field(default_factory=list)
    clarification_responses: List[ClarificationResponse] = Field(default_factory=list)
    
    # Research planning
    research_brief: Optional[ResearchBrief] = None
    research_tasks: List[ResearchTask] = Field(default_factory=list)
    
    # Research execution
    research_results: List[ResearchResult] = Field(default_factory=list)
    compressed_research: Optional[CompressedResearch] = None
    
    # Final output
    final_report: Optional[FinalReport] = None
    
    # Metadata and tracking
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_iterations: int = 0
    error_count: int = 0
    
    def add_research_result(self, result: ResearchResult) -> None:
        """Add a research result and update iteration count"""
        self.research_results.append(result)
        self.total_iterations += result.iterations_used
        
    def advance_stage(self, new_stage: WorkflowStage) -> None:
        """Advance workflow to next stage"""
        self.stage = new_stage
        if new_stage == WorkflowStage.COMPLETED:
            self.end_time = datetime.now()


# =====================================================
# Agent-Specific State Models
# =====================================================
# These models track individual agent states

class SupervisorState(BaseModel):
    """State for the Supervisor/Orchestrator agent"""
    active_researchers: List[str] = Field(default_factory=list)
    pending_tasks: List[ResearchTask] = Field(default_factory=list)
    completed_tasks: List[str] = Field(default_factory=list)
    research_iterations: int = 0
    max_iterations: int = 3
    
    def can_continue_research(self) -> bool:
        """Check if more research iterations are allowed"""
        return self.research_iterations < self.max_iterations
        

class ResearcherState(BaseModel):
    """State for individual Researcher agents"""
    agent_id: str
    current_task: Optional[ResearchTask] = None
    tool_call_iterations: int = 0
    max_tool_calls: int = 5
    collected_data: List[Dict[str, Any]] = Field(default_factory=list)
    search_queries: List[str] = Field(default_factory=list)
    
    def can_use_tools(self) -> bool:
        """Check if more tool calls are allowed"""
        return self.tool_call_iterations < self.max_tool_calls


# =====================================================
# Tool-Related Models (AutoGen Tools Pattern)
# =====================================================

class WebSearchResult(BaseModel):
    """Result from web search tool"""
    query: str
    results: List[Dict[str, Any]]
    source: str  # tavily or duckduckgo
    timestamp: datetime = Field(default_factory=datetime.now)
    

class WebPageSummary(BaseModel):
    """Summary of a web page"""
    url: str
    summary: str
    key_excerpts: List[str]
    relevance_score: float = Field(ge=0.0, le=1.0)
    

class MCPToolResult(BaseModel):
    """Result from MCP tool execution"""
    tool_name: str
    result: Any
    success: bool
    error_message: Optional[str] = None


# =====================================================
# Shared Context Models
# =====================================================
# These models are used for sharing context between agents

class SharedContext(BaseModel):
    """
    Shared context accessible by all agents.
    Replaces the Annotated[list, operator.add] pattern from LangGraph.
    """
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    raw_notes: List[str] = Field(default_factory=list)
    
    def add_message(self, message: Dict[str, Any]) -> None:
        """Add a message to shared context"""
        self.messages.append(message)
        
    def add_note(self, note: str, raw: bool = False) -> None:
        """Add a note to shared context"""
        if raw:
            self.raw_notes.append(note)
        else:
            self.notes.append(note)
            
    def get_recent_messages(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get n most recent messages"""
        return self.messages[-n:] if len(self.messages) > n else self.messages


# =====================================================
# Utility Functions
# =====================================================

def create_initial_state(workflow_id: str, user_request: str) -> ResearchState:
    """Create initial research state from user request"""
    return ResearchState(
        workflow_id=workflow_id,
        user_request=UserRequest(
            content=user_request,
            timestamp=datetime.now(),
            context=[]
        )
    )


def merge_research_results(results: List[ResearchResult]) -> Dict[str, Any]:
    """Merge multiple research results into consolidated findings"""
    merged = {
        "topics": [],
        "all_findings": [],
        "all_sources": [],
        "average_confidence": 0.0
    }
    
    for result in results:
        merged["topics"].append(result.topic)
        merged["all_findings"].append(result.findings)
        merged["all_sources"].extend(result.sources)
        merged["average_confidence"] += result.confidence
    
    if results:
        merged["average_confidence"] /= len(results)
        
    # Remove duplicates from sources
    merged["all_sources"] = list(set(merged["all_sources"]))
    
    return merged


# =====================================================
# State Persistence (for AutoGen state management)
# =====================================================

class StateManager:
    """
    Manages state persistence and recovery for the research workflow.
    Compatible with AutoGen's state management patterns.
    """
    
    def __init__(self):
        self.states: Dict[str, ResearchState] = {}
        
    def save_state(self, state: ResearchState) -> str:
        """Save state and return state ID"""
        self.states[state.workflow_id] = state
        return state.workflow_id
        
    def load_state(self, workflow_id: str) -> Optional[ResearchState]:
        """Load state by workflow ID"""
        return self.states.get(workflow_id)
        
    def update_state(self, workflow_id: str, updates: Dict[str, Any]) -> bool:
        """Update specific fields in a state"""
        if workflow_id in self.states:
            state = self.states[workflow_id]
            for key, value in updates.items():
                if hasattr(state, key):
                    setattr(state, key, value)
            return True
        return False
        
    def list_active_workflows(self) -> List[str]:
        """List all active workflow IDs"""
        return [
            wid for wid, state in self.states.items()
            if state.stage != WorkflowStage.COMPLETED
        ]


