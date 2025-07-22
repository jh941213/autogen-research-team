import asyncio
import uuid
from typing import List, Dict, Any, Optional, Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from configuration import Configuration
from datetime import datetime
import json
import logging

from autogen_core import (
    AgentId,
    MessageContext,
    TopicId,
    TypeSubscription,
    message_handler
)
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat, GraphFlow
from autogen_agentchat.messages import TextMessage, HandoffMessage
from autogen_core.tools import FunctionTool
from autogen_core.models import SystemMessage, UserMessage, AssistantMessage
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient, OpenAIChatCompletionClient

from pydantic import BaseModel, Field

from configuration import Configuration, SearchAPI, load_configuration
from state import (
    UserRequest, ClarificationRequest, ClarificationResponse,
    ResearchBrief, ResearchTask, ResearchResult, CompressedResearch,
    FinalReport, ResearchState, WorkflowStage, SupervisorState,
    ResearcherState, WebSearchResult, WebPageSummary,
    create_initial_state, merge_research_results, StateManager
)
from tools import (
    get_search_tool, get_mcp_tools, web_search_tavily,
    web_search_duckduckgo, summarize_web_page, format_search_results_for_llm
)
from utils import (
    is_token_limit_exceeded, handle_api_error, generate_search_queries,
    compress_message_history, get_model_token_limit, TokenLimitError,
    validate_configuration, format_research_results, get_today_str,
    run_with_timeout, batch_process, estimate_tokens, is_content_filter_error,
    handle_content_filter_error
)
from prompts import (
    SUPERVISOR_INSTRUCTIONS, RESEARCHER_INSTRUCTIONS,
    clarify_with_user_instructions, transform_messages_into_research_topic_prompt,
    compress_research_system_prompt, final_report_generation_prompt,
    summarize_webpage_prompt, get_today_str as prompts_get_today_str
)


# 도구 호출을 위한 구조화된 출력 모델
class ConductResearch(BaseModel):
    """연구자에게 연구 작업을 할당하는 도구"""
    research_topic: str = Field(
        description="연구 주제. 최소 한 단락의 상세한 설명이 있는 단일 주제여야 합니다."
    )


class ResearchComplete(BaseModel):
    """연구 완료를 신호하는 도구"""
    pass


class ClarifyWithUser(BaseModel):
    """사용자에게 명확화를 요청하는 도구"""
    need_clarification: bool = Field(
        description="사용자로부터 명확화가 필요한지 여부"
    )
    question: str = Field(
        description="사용자에게 물어볼 명확화 질문"
    )
    verification: str = Field(
        description="사용자가 정보를 제공한 후 확인 메시지"
    )


class ResearchQuestion(BaseModel):
    """구조화된 연구 질문/브리프"""
    research_brief: str = Field(
        description="연구를 안내하는 연구 질문 또는 브리프"
    )


class ClarificationAgent(AssistantAgent):
    """연구 시작 전 사용자 요청을 명확화하는 담당 에이전트"""
    
    def __init__(self, model_client: Union[AzureOpenAIChatCompletionClient, OpenAIChatCompletionClient]):
        # 기본 시스템 메시지 - 상세한 명확화 지침은 런타임에 사용됩니다
        system_message = """당신은 연구 명확화 전문가입니다. 사용자의 연구 요청을 분석하여 명확화가 필요한지 판단하고, 필요시 추가 질문을 합니다.

약어, 줄임말 또는 알 수 없는 용어가 있으면 사용자에게 명확히 해달라고 요청하세요.
메시지 기록에서 이미 명확화 질문을 했다면, 거의 항상 다른 질문을 할 필요가 없습니다.

오늘 날짜: {}""".format(get_today_str())

        super().__init__(
            name="ClarificationAgent",
            description="사용자와 연구 요청을 명확화합니다",
            model_client=model_client,
            system_message=system_message
        )


class ResearchBriefAgent(AssistantAgent):
    """사용자 요청으로부터 연구 브리프를 생성하는 담당 에이전트"""
    
    def __init__(self, model_client: Union[AzureOpenAIChatCompletionClient, OpenAIChatCompletionClient]):
        # 기본 시스템 메시지 - 상세한 변환 지침은 런타임에 사용됩니다
        system_message = """당신은 연구 개요 전문가입니다. 사용자의 요청을 구체적이고 상세한 연구 질문으로 변환합니다.

구체성과 세부사항을 극대화하고, 알려진 모든 사용자 선호도를 포함하세요.
명시되지 않은 차원은 개방형으로 처리하고, 근거 없는 가정은 피하세요.
1인칭으로 사용자의 관점에서 요청을 표현하세요.

오늘 날짜: {}""".format(get_today_str())

        super().__init__(
            name="ResearchBriefAgent",
            description="구조화된 연구 브리프를 생성합니다",
            model_client=model_client,
            system_message=system_message
        )


class SupervisorAgent(AssistantAgent):
    """연구 워크플로우를 관리하는 오케스트레이터 에이전트"""
    
    def __init__(self, 
                 model_client: Union[AzureOpenAIChatCompletionClient, OpenAIChatCompletionClient],
                 max_iterations: int = 3,
                 max_concurrent_units: int = 5,
                 config: Optional['Configuration'] = None):
        
        # 감독자를 위한 도구 생성
        conduct_research_tool = FunctionTool(
            func=self._conduct_research,
            description="연구자에게 연구 주제를 할당합니다",
            name="conduct_research"
        )
        
        research_complete_tool = FunctionTool(
            func=self._research_complete,
            description="연구가 완료되었음을 신호합니다",
            name="research_complete"
        )
        
        system_message = SUPERVISOR_INSTRUCTIONS.format(today=get_today_str())

        super().__init__(
            name="SupervisorAgent",
            description="연구 프로세스를 오케스트레이션합니다",
            model_client=model_client,
            system_message=system_message,
            tools=[conduct_research_tool, research_complete_tool]
        )
        
        # 인스턴스 변수로 모델 클라이언트 저장
        self.model_client = model_client
        self.config = config  # config 참조 저장
        self.max_iterations = max_iterations
        self.max_concurrent_units = max_concurrent_units
        self.current_iteration = 0
        self.research_tasks = []
        
    async def _conduct_research(self, research_topic: str) -> str:
        """연구 작업 할당을 처리하는 내부 메서드"""
        task = ResearchTask(
            topic=research_topic,
            max_iterations=5,
            assigned_to=f"researcher_{len(self.research_tasks)}"
        )
        self.research_tasks.append(task)
        return f"연구 작업 할당 완료: {research_topic}"
        
    async def _research_complete(self) -> str:
        """연구 완료를 처리하는 내부 메서드"""
        return "연구가 완료로 표시되었습니다"
    
    async def create_research_tasks(self, research_brief: str, num_units: int) -> List[ResearchTask]:
        """
        연구 브리프로부터 연구 작업들을 생성합니다.
        open_deep_research 방식의 지능적 작업 분해를 기존 구조에 적용합니다.
        """
        # 시스템 프롬프트 - open_deep_research의 lead_researcher_prompt 스타일 적용
        system_prompt = f"""당신은 수석 연구원으로서 연구 프로젝트를 계획하고 관리하는 역할을 합니다.

주어진 연구 브리프를 분석하여 최대 {num_units}개의 독립적이고 병렬 수행 가능한 연구 작업으로 분해해야 합니다.

각 연구 작업은:
1. 명확하고 구체적인 주제여야 함
2. 독립적으로 수행 가능해야 함
3. 전체 연구 목표에 기여해야 함
4. 최소 한 문단 이상의 상세한 설명 포함

연구 작업이 1개면 충분한 경우 1개만 생성하고, 복잡한 주제인 경우에만 여러 개로 분해하세요.

각 작업을 다음 형식으로 제시하세요:
"연구 작업 N: [구체적인 주제와 상세 설명]"
"""

        user_prompt = f"""연구 브리프: {research_brief}

위 연구 주제를 분석하여 적절한 수의 연구 작업(최대 {num_units}개)으로 분해해주세요."""

        messages = [
            SystemMessage(content=system_prompt, source="system"),
            UserMessage(content=user_prompt, source="user")
        ]

        try:
            # 도구 없는 텍스트 생성을 위해 항상 깨끗한 클라이언트 사용
            # (parallel_tool_calls 오류 방지)
            if self.config:
                clean_client = self.config.research_model.to_client()
                response = await clean_client.create(messages)
            else:
                # config가 없는 경우 기존 클라이언트 사용 (fallback)
                response = await self.model_client.create(messages)
                    
            content = response.content
            
            tasks = []
            if isinstance(content, str):
                # 응답에서 연구 작업들 추출
                task_lines = []
                for line in content.split('\n'):
                    line = line.strip()
                    if line and (line.startswith("연구 작업") or line.startswith("**연구 작업")):
                        task_lines.append(line)
                
                # 각 작업 라인에서 ResearchTask 생성
                for i, task_line in enumerate(task_lines[:min(num_units, self.max_concurrent_units)]):
                    # "연구 작업 N:" 부분 제거하고 실제 주제 추출
                    if ":" in task_line:
                        topic = task_line.split(":", 1)[1].strip()
                        # markdown 형식 제거
                        topic = topic.replace("**", "").replace("*", "")
                    else:
                        topic = task_line.replace("**", "").replace("*", "")
                    
                    if topic:  # 빈 주제가 아닌 경우만 추가
                        task = ResearchTask(
                            topic=topic,
                            subtopics=[],
                            max_iterations=5,
                            assigned_to=f"researcher_{i}"
                        )
                        tasks.append(task)
                
                if tasks:
                    logging.info(f"AI가 생성한 {len(tasks)}개의 연구 작업: {[t.topic[:50] for t in tasks]}")
                    return tasks
                    
        except Exception as e:
            logging.warning(f"AI 기반 작업 생성 실패, 기본 방식 사용: {e}")
        
        # AI 실패 시 기본 작업 분해 방식 사용
        return self._create_fallback_tasks(research_brief, num_units)
    
    def _create_fallback_tasks(self, research_brief: str, num_units: int) -> List[ResearchTask]:
        """AI 기반 작업 생성 실패 시 사용하는 기본 작업 분해 방식"""
        tasks = []
        
        # 연구의 다양한 관점을 기반으로 작업 생성
        research_aspects = [
            f"현황 및 배경 분석: {research_brief}",
            f"기술적/방법론적 접근: {research_brief}",
            f"비교 분석 및 사례 연구: {research_brief}",
            f"미래 전망 및 발전 방향: {research_brief}",
            f"실무 적용 방안: {research_brief}"
        ]
        
        num_tasks = min(num_units, self.max_concurrent_units, len(research_aspects))
        
        for i in range(num_tasks):
            task = ResearchTask(
                topic=research_aspects[i],
                subtopics=[],
                max_iterations=5,
                assigned_to=f"researcher_{i}"
            )
            tasks.append(task)
        
        logging.info(f"기본 방식으로 {len(tasks)}개의 연구 작업 생성")
        return tasks


class ResearcherAgent(AssistantAgent):
    """특정 연구 작업을 수행하는 개별 연구자 에이전트"""
    
    def __init__(self,
                 name: str,
                 model_client: Union[AzureOpenAIChatCompletionClient, OpenAIChatCompletionClient],
                 search_tool: Optional[FunctionTool] = None,
                 mcp_tools: Optional[List[FunctionTool]] = None,
                 max_iterations: int = 5):
        
        tools = []
        
        # 검색 수행 추적을 위한 래핑된 검색 도구 생성
        if search_tool:
            wrapped_search_tool = FunctionTool(
                func=self._wrapped_web_search,
                description="관련 정보를 웹에서 검색합니다 (연구 완료 전 필수)",
                name="web_search"
            )
            tools.append(wrapped_search_tool)
            self.original_search_tool = search_tool
        else:
            self.original_search_tool = None
            
        # MCP 도구 추가
        if mcp_tools:
            tools.extend(mcp_tools)
            
        # 연구 완료 도구 추가
        research_complete_tool = FunctionTool(
            func=self._research_complete,
            description="연구 완료를 신호합니다 (웹 검색 수행 후에만)",
            name="research_complete"
        )
        tools.append(research_complete_tool)
            
        system_message = RESEARCHER_INSTRUCTIONS.format(
            research_topic="{research_topic}",
            number_of_queries=3,
            today=get_today_str()
        )

        super().__init__(
            name=name,
            description=f"연구자 에이전트: {name}",
            model_client=model_client,
            system_message=system_message,
            tools=tools
        )
        
        self.max_iterations = max_iterations
        self.current_iterations = 0
        self.research_notes = []
        
    async def _wrapped_web_search(self, query: str, max_results: int = 5) -> str:
        """검색 활동을 추적하는 래핑된 웹 검색 함수"""
        try:
            if not self.original_search_tool:
                return "검색 도구가 설정되지 않았습니다."
            
            # 원래 검색 함수 호출
            if self.original_search_tool.name in ["web_search_tavily", "web_search_duckduckgo", "web_search"]:
                # Tavily 또는 DuckDuckGo 검색 수행
                from tools import web_search_tavily, web_search_duckduckgo
                
                # 도구 이름에 따라 적절한 검색 함수 선택
                try:
                    if self.original_search_tool.name == "web_search_tavily":
                        result = await web_search_tavily(query, max_results)
                    elif self.original_search_tool.name == "web_search_duckduckgo":
                        result = await web_search_duckduckgo(query, max_results)
                    else:
                        # 기본 fallback 로직
                        import os
                        if os.getenv("TAVILY_API_KEY"):
                            result = await web_search_tavily(query, max_results)
                        else:
                            result = await web_search_duckduckgo(query, max_results)
                except Exception:
                    result = await web_search_duckduckgo(query, max_results)
                
                # 검색 결과를 research_notes에 추가
                search_note = {
                    "query": query,
                    "results_count": len(result.results),
                    "source": result.source,
                    "timestamp": result.timestamp.isoformat()
                }
                self.research_notes.append(search_note)
                
                # 결과 포맷팅
                formatted_results = f"검색 쿼리: '{query}'\n"
                formatted_results += f"검색 소스: {result.source}\n"
                formatted_results += f"결과 수: {len(result.results)}\n\n"
                
                for i, item in enumerate(result.results[:max_results], 1):
                    title = item.get('title', '제목 없음')
                    url = item.get('url', 'URL 없음')
                    snippet = item.get('snippet', '내용 없음')
                    
                    formatted_results += f"{i}. {title}\n"
                    formatted_results += f"   URL: {url}\n"
                    formatted_results += f"   내용: {snippet[:200]}...\n\n"
                
                return formatted_results
            
        except Exception as e:
            error_note = {
                "query": query,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.research_notes.append(error_note)
            return f"검색 중 오류 발생: {str(e)}"
        
    async def _research_complete(self) -> str:
        """연구 완료를 신호하는 내부 메서드"""
        # 검증: 최소한의 검색이 수행되었는지 확인
        if len(self.research_notes) == 0:
            return "❌ 연구를 완료하기 전에 먼저 웹 검색을 수행해야 합니다! web_search 도구를 사용하여 정보를 수집하세요."
        
        # 최소 2번의 검색이 권장됨
        if len(self.research_notes) < 2:
            return "⚠️ 더 포괄적인 연구를 위해 추가 검색을 권장합니다. 다른 키워드로 web_search를 한 번 더 수행하세요."
        
        # 검색 활동 요약
        total_results = sum(note.get('results_count', 0) for note in self.research_notes if 'results_count' in note)
        search_queries = [note.get('query', '') for note in self.research_notes if 'query' in note]
        
        summary = f"✅ 연구 완료! 수행된 검색: {len(self.research_notes)}회, 총 결과: {total_results}개"
        summary += f"\n검색 쿼리: {', '.join(search_queries[:3])}{'...' if len(search_queries) > 3 else ''}"
        
        return summary


class CompressionAgent(AssistantAgent):
    """연구 결과를 압축하고 종합하는 담당 에이전트"""
    
    def __init__(self, model_client: Union[AzureOpenAIChatCompletionClient, OpenAIChatCompletionClient]):
        # prompts.py에서 상세한 압축 지침 사용
        system_message = compress_research_system_prompt.format(
            date=get_today_str()
        )

        super().__init__(
            name="CompressionAgent",
            description="연구 결과를 압축하고 종합합니다",
            model_client=model_client,
            system_message=system_message
        )


class ReportWriterAgent(AssistantAgent):
    """최종 연구 보고서를 생성하는 담당 에이전트"""
    
    def __init__(self, model_client: Union[AzureOpenAIChatCompletionClient, OpenAIChatCompletionClient]):
        # 기본 시스템 메시지 - 상세한 보고서 생성 지침은 런타임에 사용됩니다
        system_message = """당신은 전문 연구 보고서 작성자입니다. 압축된 연구 결과를 포괄적이고 잘 구조화된 최종 보고서로 변환합니다.

다음 요구사항을 따르세요:
- 적절한 제목으로 잘 구성 (제목은 #, 섹션은 ##, 하위 섹션은 ###)
- 연구에서 얻은 구체적인 사실과 통찰력 포함
- [제목](URL) 형식으로 관련 출처 참조
- 균형 잡히고 철저한 분석 제공
- 모든 참조 링크가 포함된 "출처" 섹션을 끝에 포함
- 마크다운 형식 사용

오늘 날짜: {}""".format(get_today_str())

        super().__init__(
            name="ReportWriterAgent",
            description="최종 연구 보고서를 생성합니다",
            model_client=model_client,
            system_message=system_message
        )


class DeepResearchTeam:
    """
    딥 리서치 팀 클래스
    - AI 기반 연구 완료 판정 시스템을 사용하는 AutoGen 기반 연구팀
    - 하드코딩된 임계값 대신 SupervisorAgent가 연구 충분성을 지능적으로 판단
    
    AutoGen Teams를 사용하는 딥 리서치 워크플로우의 메인 오케스트레이터
    """
    
    def __init__(self, config: Configuration):
        # 초기화 - 설정을 바탕으로 모든 에이전트와 도구들을 설정
        self.config = config
        self.state_manager = StateManager()
        
        # 설정 검증
        validation_messages = validate_configuration(config)
        for msg in validation_messages:
            logging.warning(msg)
        
        # 모델 클라이언트 가져오기
        self.model_clients = config.get_model_clients()
        
        # 에이전트 초기화
        self._initialize_agents()
        
        # 도구 초기화
        self._initialize_tools()
        
        # 워크플로우 팀 생성
        self._create_workflow_team()
        
    def _initialize_agents(self):
        """각 에이전트를 해당 모델 클라이언트로 초기화합니다"""
        # 핵심 워크플로우 에이전트들 (도구 없음 - 일반 클라이언트 사용)
        self.clarification_agent = ClarificationAgent(
            model_client=self.model_clients["research"]
        )
        
        self.brief_agent = ResearchBriefAgent(
            model_client=self.model_clients["research"]
        )
        
        # 감독자 에이전트 (일반 클라이언트 사용, 내부에서 도구 사용 시 별도 클라이언트 생성)
        self.supervisor_agent = SupervisorAgent(
            model_client=self.model_clients["research"],
            max_iterations=self.config.max_researcher_iterations,
            max_concurrent_units=self.config.max_concurrent_research_units,
            config=self.config
        )
        
        # 표준 압축 에이전트 사용
        self.compression_agent = CompressionAgent(
            model_client=self.model_clients["compression"]
        )
        
        self.report_writer_agent = ReportWriterAgent(
            model_client=self.model_clients["final_report"]
        )
        
    def _initialize_tools(self):
        """검색 및 MCP 도구들을 초기화합니다"""
        # 설정에 따라 검색 도구 가져오기 (다중 API 지원)
        from tools import get_search_tools
        self.search_tools = get_search_tools(self.config)
        self.search_tool = self.search_tools[0] if self.search_tools else None
        
        # 설정된 경우 MCP 도구 가져오기
        self.mcp_tools = []
        if self.config.mcp_config:
            self.mcp_tools = get_mcp_tools(self.config.mcp_config)
            
        # 연구자 에이전트 풀 생성 (도구 사용 - 도구 지원 클라이언트 사용)
        self.researcher_agents = []
        for i in range(self.config.max_concurrent_research_units):
            researcher = ResearcherAgent(
                name=f"Researcher_{i}",
                model_client=self.config.get_model_client_with_tools("research"),
                search_tool=self.search_tool,
                mcp_tools=self.mcp_tools,
                max_iterations=self.config.max_react_tool_calls
            )
            self.researcher_agents.append(researcher)
            
    def _create_workflow_team(self):
        """SelectorGroupChat 패턴을 사용하여 메인 워크플로우 팀을 생성합니다"""
        # 현재는 단순화된 순차적 워크플로우를 사용
        
        self.all_agents = [
            self.clarification_agent,
            self.brief_agent,
            self.supervisor_agent,
            *self.researcher_agents,
            self.compression_agent,
            self.report_writer_agent
        ]
        
    async def process_research_request(self, user_request: str) -> FinalReport:
        # 메인 연구 처리 메서드 - 전체 워크플로우를 관리하는 핵심 함수
        """전체 워크플로우를 통해 연구 요청을 처리합니다"""
        # 초기 상태 생성
        workflow_id = str(uuid.uuid4())
        state = create_initial_state(workflow_id, user_request)
        self.state_manager.save_state(state)
        
        try:
            # 1단계: 명확화 (활성화된 경우)
            if self.config.allow_clarification:
                state.advance_stage(WorkflowStage.CLARIFICATION)
                clarification_result = await self._clarify_request(state)
                if clarification_result and clarification_result.need_clarification:
                    state.clarification_requests.append(clarification_result)
                    # 실제 구현에서는 사용자 응답을 기다릴 것
                    # 현재는 원래 요청으로 계속 진행
                    
            # 2단계: 연구 브리프 생성
            state.advance_stage(WorkflowStage.PLANNING)
            research_brief = await self._create_research_brief(state)
            state.research_brief = research_brief
            
            # 3단계: 연구 수행 (감독자 반복과 함께)
            state.advance_stage(WorkflowStage.RESEARCH)
            
            supervisor_state = SupervisorState(
                active_researchers=[],
                pending_tasks=[],
                completed_tasks=[],
                research_iterations=0,
                max_iterations=self.config.max_researcher_iterations
            )
            
            # 감독자 감시 하에 연구 루프
            while supervisor_state.can_continue_research():
                # 감독자가 연구 작업 생성
                tasks = await self._supervisor_plan_research(state, supervisor_state)
                
                # 병렬로 연구 실행
                research_results = await self._conduct_parallel_research(tasks, state)
                
                # 결과를 상태에 추가
                for result in research_results:
                    state.add_research_result(result)
                
                supervisor_state.research_iterations += 1
                
                # 추가 연구가 필요한지 확인
                if await self._is_research_sufficient(state, supervisor_state):
                    break
            
            # 4단계: 연구 압축
            state.advance_stage(WorkflowStage.COMPRESSION)
            compressed_research = await self._compress_research(state)
            state.compressed_research = compressed_research
            
            # 5단계: 최종 보고서 생성
            state.advance_stage(WorkflowStage.REPORT_GENERATION)
            final_report = await self._generate_report(state)
            state.final_report = final_report
            
            # 완료로 표시
            state.advance_stage(WorkflowStage.COMPLETED)
            self.state_manager.save_state(state)
            
            return final_report
            
        except TokenLimitError as e:
            logging.error(f"토큰 한계 초과: {e}")
            # 컨텍스트 압축으로 복구 시도
            state = await self._handle_token_limit(state)
            return await self.process_research_request(user_request)
            
        except Exception as e:
            state.error_count += 1
            self.state_manager.save_state(state)
            logging.error(f"연구 오류: {e}")
            raise e
            
    async def _clarify_request(self, state: ResearchState) -> Optional[ClarificationRequest]:
        """명확화 단계를 처리합니다"""
        messages = [
            TextMessage(
                content=f"이 연구 요청을 분석하고 명확화가 필요한지 판단하세요: {state.user_request.content}",
                source="system"
            )
        ]
        
        try:
            response = await self.clarification_agent.on_messages(messages, None)
            
            # 실제 구현에서는 구조화된 출력을 파싱할 것
            # 현재는 응답에서 명확화가 언급되는지 확인
            response_content = response.chat_message.content if response.chat_message else ""
            
            if "명확화" in response_content or "clarification" in response_content.lower() and "needed" in response_content.lower():
                return ClarificationRequest(
                    need_clarification=True,
                    question=response_content,
                    verification="귀하의 명확화에 기반하여 연구를 진행하겠습니다.",
                    original_request=state.user_request.content
                )
            
            return None
            
        except Exception as e:
            logging.error(f"명확화 실패: {e}")
            return None
        
    async def _create_research_brief(self, state: ResearchState) -> ResearchBrief:
        """사용자 요청으로부터 연구 브리프를 생성합니다"""
        messages = [
            TextMessage(
                content=f"다음에 대한 포괄적인 연구 브리프를 생성하세요: {state.user_request.content}",
                source="system"
            )
        ]
        
        response = await self.brief_agent.on_messages(messages, None)
        
        brief = ResearchBrief(
            research_brief=response.chat_message.content,
            original_request=state.user_request.content,
            clarifications=[]
        )
        
        return brief
        
    async def _supervisor_plan_research(self, 
                                       state: ResearchState, 
                                       supervisor_state: SupervisorState) -> List[ResearchTask]:
        """감독자가 연구 작업들을 계획합니다"""
        # 감독자를 위한 컨텍스트 생성
        context = f"""
연구 브리프: {state.research_brief.research_brief}

이전 연구 결과: {len(state.research_results)}개
현재 반복: {supervisor_state.research_iterations + 1} / {supervisor_state.max_iterations}

다음 연구 작업들을 계획하세요. 주제를 구체적이고 집중된 연구 단위로 분해하세요.
"""
        
        messages = [
            TextMessage(content=context, source="system")
        ]
        
        # 감독자의 계획 가져오기
        response = await self.supervisor_agent.on_messages(messages, None)
        
        # 감독자 응답에서 연구 작업 추출
        # AI를 사용하여 지능적으로 작업 분해 (open_deep_research 방식 적용)
        tasks = await self.supervisor_agent.create_research_tasks(
            state.research_brief.research_brief,
            min(
                len(self.researcher_agents), 
                self.config.max_concurrent_research_units
            )
        )
        
        supervisor_state.pending_tasks.extend(tasks)
        return tasks
        
    async def _conduct_parallel_research(self, 
                                       tasks: List[ResearchTask],
                                       state: ResearchState) -> List[ResearchResult]:
        # 병렬 연구 수행 - 여러 연구자가 동시에 다른 주제를 연구
        """적절한 오류 처리와 함께 연구 작업들을 병렬로 수행합니다"""
        # 속도 제한을 위한 세마포어 생성
        semaphore = asyncio.Semaphore(self.config.max_concurrent_research_units)
        
        async def conduct_single_research(researcher: ResearcherAgent, 
                                        task: ResearchTask) -> Optional[ResearchResult]:
            """오류 처리와 함께 단일 연구 작업을 실행합니다"""
            async with semaphore:
                try:
                    # 연구자의 시스템 메시지를 특정 주제로 업데이트
                    researcher_state = ResearcherState(
                        agent_id=researcher.name,
                        current_task=task,
                        tool_call_iterations=0,
                        max_tool_calls=self.config.max_react_tool_calls,
                        collected_data=[],
                        search_queries=[]
                    )
                    
                    # 집중된 연구 프롬프트 생성
                    research_prompt = f"""
연구 주제: {task.topic}

귀하의 임무는 이 특정 주제에 대해 철저한 연구를 수행하는 것입니다.
사용 가능한 검색 도구를 사용하여 관련성 높고 고품질의 정보를 찾으세요.
다음에 집중하세요:
1. 최신 개발 동향 및 트렌드
2. 핵심 데이터 및 통계
3. 전문가 의견 및 분석
4. 실용적 의미

주제의 다양한 측면을 탐구하기 위해 다양한 검색 쿼리를 생성하세요.
"""
                    
                    messages = [
                        TextMessage(content=research_prompt, source="system")
                    ]
                    
                    # 타임아웃과 함께 표준 연구 실행
                    research_response = await run_with_timeout(
                        researcher.on_messages(messages, None),
                        timeout_seconds=300  # 연구 작업당 5분 타임아웃
                    )
                        
                    if not research_response:
                        logging.warning(f"연구 타임아웃 주제: {task.topic}")
                        return None
                        
                    # 응답에서 연구 결과 추출
                    findings = research_response.chat_message.content if research_response.chat_message else ""
                    
                    # 검색 노트에서 소스 URL 추출
                    sources = []
                    for note in researcher.research_notes:
                        if 'query' in note:
                            # 실제 검색 결과에서 URL들을 추출해야 하지만, 
                            # 현재는 기본적인 소스 정보 포함
                            sources.append(f"검색: '{note['query']}' ({note.get('source', '알 수 없음')})")
                        
                    # 연구 결과 생성
                    result = ResearchResult(
                        topic=task.topic,
                        findings=findings,
                        sources=sources,  # 실제 검색 소스들 포함
                        raw_notes=researcher.research_notes,  # 검색 노트 포함
                        confidence=0.8 if sources else 0.3,  # 검색 여부에 따라 신뢰도 조정
                        researcher_id=researcher.name,
                        iterations_used=researcher_state.tool_call_iterations
                    )
                        
                    return result
                    
                except TokenLimitError as e:
                    logging.error(f"Token limit in research: {e}")
                    # Try with reduced context
                    return await self._conduct_research_with_reduced_context(
                        researcher, task, researcher_state
                    )
                    
                except Exception as e:
                    logging.error(f"Research error for {task.topic}: {e}")
                    
                    # Azure OpenAI 콘텐츠 필터 에러 처리
                    if is_content_filter_error(e):
                        return handle_content_filter_error(task.topic, researcher.name, state.user_request.content)
                    
                    # Check if we should retry for other errors
                    if await handle_api_error(e):
                        return await conduct_single_research(researcher, task)
                    return None
        
        # Assign tasks to researchers
        research_coroutines = []
        for i, task in enumerate(tasks[:len(self.researcher_agents)]):
            researcher = self.researcher_agents[i]
            coroutine = conduct_single_research(researcher, task)
            research_coroutines.append(coroutine)
        
        # Execute all research in parallel
        results = await asyncio.gather(*research_coroutines, return_exceptions=False)
        
        # Filter out None results
        valid_results = [r for r in results if r is not None]
        
        return valid_results
        
    async def _conduct_research_with_reduced_context(self,
                                                   researcher: ResearcherAgent,
                                                   task: ResearchTask,
                                                   researcher_state: ResearcherState) -> Optional[ResearchResult]:
        """Conduct research with reduced context after token limit error"""
        try:
            # Create minimal research prompt
            minimal_prompt = f"Research this topic concisely: {task.topic[:200]}"
            
            messages = [
                TextMessage(content=minimal_prompt, source="system")
            ]
            
            response = await researcher.on_messages(messages, None)
            
            return ResearchResult(
                topic=task.topic,
                findings=response.chat_message.content if response.chat_message else "",
                sources=[],
                raw_notes=[],
                confidence=0.6,  # Lower confidence due to reduced context
                researcher_id=researcher.name,
                iterations_used=1
            )
            
        except Exception as e:
            logging.error(f"Failed even with reduced context: {e}")
            return None
        
    async def _is_research_sufficient(self, 
                                    state: ResearchState,
                                    supervisor_state: SupervisorState) -> bool:
        # AI 기반 연구 충분성 판단 - 핵심 개선 부분!
        # 기존 하드코딩된 조건(avg_confidence >= 0.8, unique_sources >= 5) 대신
        # SupervisorAgent가 연구 내용을 분석하여 지능적으로 완료 여부를 결정
        """Determine if collected research is sufficient using AI-based decision making"""
        # Hard limit: maximum iterations reached
        if supervisor_state.research_iterations >= supervisor_state.max_iterations:
            return True
            
        # Minimum threshold: need at least some research results
        if len(state.research_results) == 0:
            return False
            
        # 감독자 에이전트를 사용하여 AI 기반 결정
        try:
            # 감독자 평가를 위한 연구 요약 준비
            research_summary = self._create_research_summary_for_evaluation(state)
            
            # 감독자를 위한 평가 컨텍스트 생성
            evaluation_context = f"""
연구 평가 작업

연구 주제: {state.research_brief.research_brief if state.research_brief else state.user_request.content}

현재 연구 결과 요약:
{research_summary}

현재 반복: {supervisor_state.research_iterations} / {supervisor_state.max_iterations}
완료된 총 연구 단위: {len(state.research_results)}개

평가 지침:
연구 감독자로서, 현재 연구가 사용자의 질문에 답하는 포괄적인 보고서를 작성하기에 충분한지 결정해야 합니다.

결정 기준:
다음 조건이 모두 충족되면 "research_complete" 도구를 사용하세요:
- 주요 질문의 핵심 측면과 하위 주제들이 철저히 다뤄졌음
- 여러 관점에서 충분한 깊이와 폭의 정보가 있음
- 신뢰할 수 있고 다양한 소스들이 참조됨 (5개 이상의 고유 소스 목표)
- 좋은 신뢰도 점수로 정보 품질이 높음 (평균 >0.7)
- 포괄적인 보고서 작성을 방해하는 중요한 공백이 남아있지 않음

다음 조건 중 하나라도 해당되면 "conduct_research" 도구를 사용하세요:
- 중요한 측면이나 하위 주제가 누락되거나 충분히 탐구되지 않음
- 소스 다양성이 부족함 (4-5개 미만의 고유 소스)
- 핵심 요점에 대한 충분한 깊이나 세부사항이 부족함
- 낮은 신뢰도 점수 (평균 <0.6)로 불확실한 정보를 나타냄
- 최종 보고서를 약화시킬 수 있는 중요한 공백이 존재

중요: 평가에 기반하여 "research_complete" 또는 "conduct_research" 도구 중 하나를 반드시 호출해야 합니다. 단순한 텍스트 분석만 제공하지 말고 - 적절한 도구를 사용하여 결정을 신호하세요.

지금 평가를 수행하고 적절한 도구를 호출하세요.
"""

            messages = [
                TextMessage(content=evaluation_context, source="system")
            ]
            
            # 도구 사용을 통한 감독자의 결정 가져오기
            response = await self.supervisor_agent.on_messages(messages, None)
            
            # 감독자가 research_complete 도구를 호출했는지 확인
            if hasattr(response, 'chat_message') and response.chat_message:
                # 응답에서 도구 호출 확인
                if hasattr(response.chat_message, 'tool_calls') and response.chat_message.tool_calls:
                    # research_complete 도구 호출 찾기
                    for tool_call in response.chat_message.tool_calls:
                        tool_name = None
                        if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'name'):
                            tool_name = tool_call.function.name
                        elif hasattr(tool_call, 'name'):  # 대체 구조
                            tool_name = tool_call.name
                        
                        if tool_name == "research_complete":
                            logging.info("감독자 에이전트가 research_complete 호출 - 연구가 충분함")
                            return True
                        elif tool_name == "conduct_research":
                            logging.info("감독자 에이전트가 conduct_research 호출 - 연구 계속")
                            return False
                
                # Fallback: analyze content if no clear tool calls
                response_content = response.chat_message.content.lower() if response.chat_message.content else ""
                
                # Look for explicit decision indicators
                if any(phrase in response_content for phrase in ["research is complete", "sufficient research", "comprehensive enough"]):
                    logging.info("Supervisor indicated research is sufficient based on content analysis")
                    return True
                elif any(phrase in response_content for phrase in ["need more research", "insufficient", "continue research", "gaps remain"]):
                    logging.info("Supervisor indicated more research is needed based on content analysis")
                    return False
            
            # Check if tool was actually called (from logs) - backup detection
            try:
                # If we reach here, try to inspect the response more deeply
                response_str = str(response)
                if "research_complete" in response_str:
                    logging.info("Detected research_complete in response structure - research is sufficient")
                    return True
                elif "conduct_research" in response_str:
                    logging.info("Detected conduct_research in response structure - continuing research") 
                    return False
            except:
                pass
            
            # Default: continue research if decision is unclear
            logging.warning("Supervisor decision unclear - defaulting to continue research")
            return False
            
        except Exception as e:
            logging.error(f"AI-based research evaluation failed: {e}")
            # Fallback to simple threshold-based decision
            return len(state.research_results) >= 3 and supervisor_state.research_iterations >= 2
    
    def _create_research_summary_for_evaluation(self, state: ResearchState) -> str:
        """감독자 평가를 위한 현재 연구의 간결한 요약을 생성합니다"""
        if not state.research_results:
            return "아직 사용 가능한 연구 결과가 없습니다."
        
        summary_parts = []
        
        for i, result in enumerate(state.research_results, 1):
            summary_parts.append(f"""
연구 단위 {i}:
주제: {result.topic}
신뢰도: {result.confidence:.2f}
소스: {len(result.sources)}개
핵심 결과: {result.findings[:300]}{'...' if len(result.findings) > 300 else ''}
""")
        
        # 집계 통계 추가
        total_confidence = sum(r.confidence for r in state.research_results)
        avg_confidence = total_confidence / len(state.research_results)
        
        merged_results = merge_research_results(state.research_results)
        unique_sources = len(merged_results["all_sources"])
        
        aggregate_info = f"""
집계 통계:
- 총 연구 단위: {len(state.research_results)}개
- 평균 신뢰도: {avg_confidence:.2f}
- 고유 소스: {unique_sources}개
- 다룬 주제: {', '.join(merged_results['topics'])}
"""
        
        return aggregate_info + "\n".join(summary_parts)
        
    async def _compress_research(self, state: ResearchState) -> CompressedResearch:
        # 연구 결과 압축 - 모든 연구 결과를 요약하고 정리
        """표준 압축을 사용하여 모든 연구 결과를 압축합니다"""
        logging.info("표준 압축 시스템 사용")
        # 모든 연구 결과 병합
        merged_data = merge_research_results(state.research_results)
        
        # 압축을 위한 결과 포맷팅
        findings_text = "\n\n".join([
            f"주제: {r.topic}\n결과: {r.findings}\n"
            for r in state.research_results
        ])
        
        # 토큰 한계 확인
        estimated_tokens = estimate_tokens(findings_text)
        max_tokens = get_model_token_limit(self.config.compression_model.model_name)
        
        if max_tokens and estimated_tokens > max_tokens * 0.8:
            # 청크로 나누어 별도 압축
            findings_text = await self._chunk_compress_findings(state.research_results)
        
        compression_prompt = f"""
다음 연구 결과들을 일관된 요약으로 압축하고 종합하세요:

{findings_text}

다음을 포함하는 포괄적인 요약을 생성하세요:
1. 핵심 주제와 패턴 식별
2. 가장 중요한 결과 강조
3. 사실적 정확성 유지
4. 소스 귀속 보존
"""
        
        messages = [
            TextMessage(content=compression_prompt, source="system")
        ]
        
        response = await self.compression_agent.on_messages(messages, None)
        
        compressed = CompressedResearch(
            summary=response.chat_message.content,
            key_findings=merged_data["topics"][:10],  # 상위 10개 주제
            methodology="교차 검증을 통한 다중 에이전트 병렬 연구",
            combined_sources=merged_data["all_sources"][:20],  # 상위 20개 소스
            topics_covered=merged_data["topics"]
        )
        
        return compressed
        
    async def _chunk_compress_findings(self, results: List[ResearchResult]) -> str:
        """Compress findings in chunks to handle token limits"""
        # Group results into chunks
        chunk_size = 3
        chunks = [results[i:i+chunk_size] for i in range(0, len(results), chunk_size)]
        
        compressed_chunks = []
        
        for chunk in chunks:
            chunk_text = "\n\n".join([
                f"Topic: {r.topic}\nFindings: {r.findings}\n"
                for r in chunk
            ])
            
            messages = [
                TextMessage(
                    content=f"Summarize these research findings concisely:\n\n{chunk_text}",
                    source="system"
                )
            ]
            
            response = await self.compression_agent.on_messages(messages, None)
            compressed_chunks.append(response.chat_message.content)
        
        # Join compressed chunks
        return "\n\n---\n\n".join(compressed_chunks)
        
    async def _generate_report(self, state: ResearchState) -> FinalReport:
        """최종 연구 보고서를 생성합니다"""
        report_prompt = f"""
        다음에 기반하여 포괄적인 연구 보고서를 생성하세요:

        원래 요청: {state.user_request.content}
        연구 브리프: {state.research_brief.research_brief}

        압축된 결과:
        {state.compressed_research.summary}

        다룬 주요 주제: {', '.join(state.compressed_research.topics_covered[:10])}

        다음 구조로 보고서를 작성하세요:
        1. 요약 (주요 내용)
        2. 서론
        3. 핵심 결과 (주제별로 정리)
        4. 분석 및 통찰
        5. 결론
        6. 추천사항
        7. 출처

        명확한 마크다운 포맷팅을 사용하세요. 포괄적이면서도 간결하게 작성하세요.
        """
                
        messages = [
            TextMessage(content=report_prompt, source="system")
        ]
        
        response = await self.report_writer_agent.on_messages(messages, None)
        
        # LLM이 생성한 전체 보고서 내용을 그대로 사용
        report_content = response.chat_message.content
        
        report = FinalReport(
            title=f"연구 보고서: {state.user_request.content}",
            executive_summary=report_content,  # LLM 전체 출력을 executive_summary에 저장
            detailed_findings=report_content,  # 필요시 같은 내용 저장 (호환성 위해)
            conclusion="",  # 빈 문자열로 설정
            recommendations=[],  # 빈 리스트로 설정
            sources=state.compressed_research.combined_sources,
            metadata={
                "workflow_id": state.workflow_id,
                "duration": (datetime.now() - state.start_time).total_seconds(),
                "total_iterations": state.total_iterations,
                "researchers_used": len(state.research_results),
                "unique_sources": len(state.compressed_research.combined_sources),
                "confidence_score": state.compressed_research.average_confidence 
                    if hasattr(state.compressed_research, 'average_confidence') else 0.0
            }
        )
        
        return report
        
    def _extract_section(self, content: str, start_marker: str, end_marker: str) -> str:
        """Extract a section from markdown content"""
        try:
            start_idx = content.lower().find(start_marker.lower())
            end_idx = content.lower().find(end_marker.lower())
            
            if start_idx == -1:
                return ""
                
            if end_idx == -1:
                return content[start_idx:]
                
            return content[start_idx:end_idx].strip()
            
        except:
            return ""
            
    def _extract_recommendations(self, content: str) -> List[str]:
        """Extract recommendations from report content"""
        try:
            rec_section = self._extract_section(content, "Recommendations", "Sources")
            if not rec_section:
                return []
                
            # Simple extraction of bullet points
            lines = rec_section.split('\n')
            recommendations = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('- ') or line.startswith('* ') or line.startswith('• '):
                    recommendations.append(line[2:].strip())
                elif line and not line.startswith('#'):
                    recommendations.append(line)
                    
            return recommendations[:10]  # Limit to 10 recommendations
            
        except:
            return []
            
    async def _handle_token_limit(self, state: ResearchState) -> ResearchState:
        """Handle token limit errors by reducing context"""
        logging.info("Handling token limit by compressing state")
        
        # Compress research results if too many
        if len(state.research_results) > 10:
            # Keep only the most relevant results
            state.research_results = sorted(
                state.research_results,
                key=lambda r: r.confidence,
                reverse=True
            )[:10]
        
        # Clear raw notes to save space
        for result in state.research_results:
            result.raw_notes = []
        
        self.state_manager.save_state(state)
        return state


# 메인 실행 함수
async def run_deep_research(
    user_request: str, 
    config: Optional[Configuration] = None
) -> FinalReport:
    """
    딥 리서치 실행을 위한 메인 진입점
    
    Args:
        user_request: 사용자의 연구 요청
        config: 선택적 설정 객체 (제공되지 않으면 기본값 사용)
        
    Returns:
        연구 결과를 포함하는 FinalReport 객체
    """
    if config is None:
        config = load_configuration()
        
    # 연구팀 생성
    research_team = DeepResearchTeam(config)
    
    # 요청 처리
    report = await research_team.process_research_request(user_request)
    
    return report



