"""
AutoGen Deep Research System
============================

AutoGen 프레임워크를 사용한 향상된 심층 연구 시스템입니다.
LangGraph의 state machine 패턴을 AutoGen의 Teams와 Agent 패턴으로 변환했습니다.

주요 구성 요소:
- DeepResearchTeam: 메인 연구 워크플로우 오케스트레이터
- ResearcherAgent: 개별 연구 수행 에이전트
- SupervisorAgent: 연구 감독 및 관리 에이전트
- Enhanced Termination System: 유연한 종료 조건 시스템
"""

from .deep_researcher import (
    DeepResearchTeam,
    ResearcherAgent, 
    SupervisorAgent,
    ClarificationAgent,
    ResearchBriefAgent,
    CompressionAgent,
    ReportWriterAgent,
    run_deep_research
)

from .configuration import (
    Configuration,
    SearchAPI,
    MCPConfig,
    load_configuration
)

from .state import (
    # Core state classes
    UserRequest,
    ClarificationRequest,
    ClarificationResponse,
    ResearchBrief,
    ResearchTask,
    ResearchResult,
    CompressedResearch,
    FinalReport,
    ResearchState,
    WorkflowStage,
    SupervisorState,
    ResearcherState,
    WebSearchResult,
    WebPageSummary,
    StateManager,
    
    # Tool classes
    ConductResearch,
    ResearchComplete,
    ClarifyWithUser,
    ResearchQuestion,
    
    # Enhanced termination system
    TerminationCondition,
    TerminationManager,
    TerminationReason,
    MaxIterationTermination,
    FunctionCallTermination,
    TextMentionTermination,
    ResearchCompleteTermination,
    ExternalTermination,
    OrTerminationCondition,
    AndTerminationCondition,
    create_default_researcher_termination,
    create_default_supervisor_termination,
    
    # Helper functions
    create_initial_state,
    merge_research_results
)

__version__ = "0.1.0"
__author__ = "AutoGen Deep Research Team"
__description__ = "Enhanced deep research system using AutoGen framework"

__all__ = [
    # Main classes
    "DeepResearchTeam",
    "ResearcherAgent",
    "SupervisorAgent",
    "ClarificationAgent", 
    "ResearchBriefAgent",
    "CompressionAgent",
    "ReportWriterAgent",
    
    # Main function
    "run_deep_research",
    
    # Configuration
    "Configuration",
    "SearchAPI",
    "MCPConfig",
    "load_configuration",
    
    # State management
    "UserRequest",
    "ClarificationRequest", 
    "ClarificationResponse",
    "ResearchBrief",
    "ResearchTask",
    "ResearchResult",
    "CompressedResearch",
    "FinalReport",
    "ResearchState",
    "WorkflowStage",
    "SupervisorState",
    "ResearcherState",
    "WebSearchResult",
    "WebPageSummary",
    "StateManager",
    
    # Tool classes
    "ConductResearch",
    "ResearchComplete", 
    "ClarifyWithUser",
    "ResearchQuestion",
    
    # Termination system
    "TerminationCondition",
    "TerminationManager",
    "TerminationReason",
    "MaxIterationTermination",
    "FunctionCallTermination",
    "TextMentionTermination", 
    "ResearchCompleteTermination",
    "ExternalTermination",
    "OrTerminationCondition",
    "AndTerminationCondition",
    "create_default_researcher_termination",
    "create_default_supervisor_termination",
    
    # Helper functions
    "create_initial_state",
    "merge_research_results"
] 