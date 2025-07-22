# AutoGen Open Deep Research 시스템

**[Microsoft AutoGen 프레임워크](https://github.com/microsoft/autogen)**를 기반으로 한 향상된 심층 연구 시스템입니다. LangGraph의 state machine 패턴을 AutoGen의 **Teams**와 **Agent** 패턴으로 변환하여, 다중 에이전트 협업을 통한 포괄적이고 체계적인 연구를 수행합니다.

> **📢 v2.1 업데이트 예정**: AutoGen **GraphFlow**를 적용한 고급 워크플로우 라우팅 시스템 구현 예정입니다. 현재는 기본 Agent 패턴과 FunctionTool 기반으로 구현되어 있습니다.

## ✨ **최신 업데이트** ([v2.0](https://github.com/jh941213/autogen-research-team/releases/tag/v2.0))

🧠 **AI 기반 지능적 작업 분해** - open_deep_research 방식 통합  
⚡ **자동 오류 처리** - Azure 콘텐츠 필터, 토큰 한계 등  
🎯 **맞춤형 연구 계획** - 연구 주제별 동적 작업 생성  
🔄 **견고한 Fallback** - AI 실패 시 체계적 기본 분해 방식  
🚀 **최적화된 성능** - 90초 내 포괄적 연구 보고서 완성

## 🚀 **AutoGen 프레임워크 활용**

본 시스템은 Microsoft AutoGen의 핵심 기능들을 활용하여 구현되었습니다:

### 🏗️ **사용된 AutoGen 컴포넌트**
- **`AssistantAgent`**: 모든 연구 에이전트의 기본 클래스
- **`FunctionTool`**: 웹 검색, MCP 통합 등의 도구 구현
- **Model Clients**: `AzureOpenAIChatCompletionClient`, `OpenAIChatCompletionClient`
- **Messages**: `TextMessage`, `HandoffMessage` 등 구조화된 메시지 시스템
- **Teams**: `SelectorGroupChat` 패턴 (향후 `GraphFlow` 확장 예정)
- **Tool Integration**: `parallel_tool_calls` 지원으로 병렬 도구 실행

### 🔧 **AutoGen 아키텍처 패턴**
```python
# 예시: 지능적 작업 분해 기능을 가진 SupervisorAgent
class SupervisorAgent(AssistantAgent):
    def __init__(self, model_client, max_iterations=3, config=None):
        tools = [
            FunctionTool(func=self._conduct_research, name="conduct_research"),
            FunctionTool(func=self._research_complete, name="research_complete")
        ]
        super().__init__(
            name="SupervisorAgent",
            model_client=model_client,
            tools=tools,
            system_message=SUPERVISOR_INSTRUCTIONS
        )
        self.config = config  # parallel_tool_calls 오류 처리용
    
    async def create_research_tasks(self, research_brief: str, num_units: int):
        """AI 기반 지능적 연구 작업 분해 (open_deep_research 방식)"""
        # AI가 연구 주제를 분석하여 맞춤형 작업 생성
        # Fallback: AI 실패 시 체계적 기본 분해 방식 사용
        pass  # 실제 구현은 deep_researcher.py 참조
```

## 🌟 주요 특징

### 🤖 다중 에이전트 시스템 (AutoGen AssistantAgent 기반)
- **명확화 에이전트** (`ClarificationAgent`): 사용자 요청의 모호한 부분을 명확히 함
- **연구 브리프 에이전트** (`ResearchBriefAgent`): 사용자 요청을 상세한 연구 계획으로 변환
- **감독자 에이전트** (`SupervisorAgent`): AutoGen `FunctionTool`을 사용하여 연구 프로세스 오케스트레이션
- **연구자 에이전트** (`ResearcherAgent`): 병렬 연구 수행, 웹 검색 도구 통합
- **압축 에이전트** (`CompressionAgent`): 연구 결과를 정리하고 종합
- **보고서 작성자** (`ReportWriterAgent`): 최종 마크다운 보고서 생성

### 🔍 다중 검색 API 지원
- **Tavily Search API**: 고품질 검색 결과 (추천)
- **DuckDuckGo Search**: 무료 검색 옵션
- **병렬 검색**: 여러 API를 동시에 사용하여 결과 품질 향상

### 🧠 AI 기반 지능적 연구 관리

#### 📋 **지능적 연구 작업 분해** 
- **동적 작업 생성**: AI가 연구 주제의 복잡성을 분석하여 적절한 수의 작업으로 분해
- **맞춤형 작업 계획**: 연구 브리프에 따라 구체적이고 독립적인 연구 작업 생성
- **병렬 최적화**: 각 작업이 독립적으로 실행 가능하도록 지능적으로 설계
- **Fallback 시스템**: AI 기반 분해 실패 시 체계적인 기본 분해 방식으로 자동 전환

#### 🎯 **AI 기반 연구 완료 판정**
- 하드코딩된 임계값 대신 SupervisorAgent가 연구 충분성을 지능적으로 판단
- 연구 내용, 소스 다양성, 신뢰도를 종합적으로 평가
- 비용 효율적인 연구 수행

### ⚙️ 유연한 모델 지원
- **Azure OpenAI**: 엔터프라이즈급 배포 지원
- **OpenAI**: 직접 API 연동
- **로컬 모델**: Ollama 등 로컬 LLM 지원

### 📊 상태 관리 시스템
- 전체 워크플로우 상태 추적
- 에러 복구 및 재시도 메커니즘
- 연구 진행상황 실시간 모니터링

## 🏗️ AutoGen 기반 시스템 아키텍처

```
📧 사용자 요청 (TextMessage)
    ↓
🤔 명확화 단계 (ClarificationAgent extends AssistantAgent)
    ↓
📋 연구 계획 생성 (ResearchBriefAgent extends AssistantAgent)
    ↓
🎯 연구 오케스트레이션 (SupervisorAgent extends AssistantAgent)
    │   ├── 🧠 AI 기반 지능적 작업 분해 (create_research_tasks)
    │   │   ├── 동적 작업 수 결정 (1~N개)
    │   │   ├── 맞춤형 작업 생성
    │   │   └── Fallback → 체계적 기본 분해
    │   ├── FunctionTool: conduct_research
    │   └── FunctionTool: research_complete
    ↓
🔍 병렬 연구 수행 (ResearcherAgent × N extends AssistantAgent)
    │   ├── FunctionTool: web_search (Tavily/DuckDuckGo)
    │   ├── FunctionTool: research_complete
    │   └── MCP Tools (선택적)
    ↓
📊 결과 압축 (CompressionAgent extends AssistantAgent)
    ↓
📄 최종 보고서 생성 (ReportWriterAgent extends AssistantAgent)
```

### 🔧 **AutoGen 통합 세부사항**

#### **Model Client 관리**
```python
# 각 에이전트별 전용 모델 클라이언트
model_clients = {
    "research": AzureOpenAIChatCompletionClient(model="gpt-4o"),
    "summarization": AzureOpenAIChatCompletionClient(model="gpt-4o-mini"),
    "compression": AzureOpenAIChatCompletionClient(model="gpt-4o-mini"),
    "final_report": AzureOpenAIChatCompletionClient(model="gpt-4o")
}

# 도구 사용 에이전트를 위한 parallel_tool_calls 활성화
tool_enabled_client = model_client.model_copy()
tool_enabled_client.parallel_tool_calls = True
```

#### **Tool Integration**
```python
# FunctionTool을 사용한 웹 검색 도구 구현
search_tool = FunctionTool(
    func=web_search_tavily,
    description="웹에서 정보를 검색합니다",
    name="web_search"
)

# 에이전트에 도구 등록
researcher = ResearcherAgent(
    name="Researcher_0",
    model_client=tool_enabled_client,
    tools=[search_tool, research_complete_tool]
)
```

## 🚀 설치 및 설정

### 1. 환경 설정

```bash
# 패키지 설치
pip install -r requirements.txt

# 환경변수 파일 생성
cp env.example .env
```

### 2. 환경변수 설정

`.env` 파일을 편집하여 다음 중 하나를 설정:

#### Option A: Azure OpenAI (추천)
```bash
MODEL_PROVIDER=azure_openai
AZURE_OPENAI_API_KEY=your-azure-openai-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
```

#### Option B: OpenAI
```bash
MODEL_PROVIDER=openai
OPENAI_API_KEY=your-openai-api-key
```

#### Option C: 로컬 모델 (Ollama)
```bash
MODEL_PROVIDER=local
LOCAL_BASE_URL=http://localhost:11434/v1
LOCAL_MODEL_NAME=llama2
```

### 3. 검색 API 설정

```bash
# Tavily (추천)
TAVILY_API_KEY=your-tavily-api-key
SEARCH_APIS=tavily,duckduckgo

# 또는 DuckDuckGo만 사용 (무료)
SEARCH_APIS=duckduckgo
```

## 📖 사용법

### 기본 사용법

```python
import asyncio
from autogen_open_deep_research import run_deep_research

async def main():
    # 연구 요청
    request = "양자 컴퓨팅의 최신 개발 동향과 암호화 분야에서의 잠재적 응용은?"
    
    # 연구 실행
    report = await run_deep_research(request)
    
    # 결과 출력
    print(f"제목: {report.title}")
    print(f"요약: {report.executive_summary}")
    print(f"출처 수: {len(report.sources)}")

# 실행
asyncio.run(main())
```

### 고급 사용법 (커스텀 설정)

```python
from autogen_open_deep_research import (
    DeepResearchTeam, 
    Configuration, 
    load_configuration
)

async def advanced_research():
    # 커스텀 설정 로드
    config = load_configuration()
    
    # 설정 수정
    config.max_concurrent_research_units = 3
    config.allow_clarification = False
    
    # 연구팀 생성
    team = DeepResearchTeam(config)
    
    # 연구 실행
    report = await team.process_research_request(
        "인공지능의 최신 동향에 대해 연구해주세요"
    )
    
    return report
```

## ⚙️ 설정 옵션

### 모델 설정

각 에이전트별로 다른 모델을 사용할 수 있습니다:

```python
# 환경변수 또는 Configuration 객체에서 설정
AZURE_OPENAI_RESEARCH_DEPLOYMENT=gpt-4o          # 메인 연구용
AZURE_OPENAI_SUMMARIZATION_DEPLOYMENT=gpt-4o-mini # 요약용
AZURE_OPENAI_COMPRESSION_DEPLOYMENT=gpt-4o-mini   # 압축용
AZURE_OPENAI_FINAL_REPORT_DEPLOYMENT=gpt-4o       # 보고서용
```

### 연구 파라미터

```python
config = Configuration()
config.max_concurrent_research_units = 5    # 동시 연구 단위 수
config.max_researcher_iterations = 3        # 최대 연구 반복 횟수
config.max_react_tool_calls = 5            # 연구자당 최대 도구 호출 수
config.allow_clarification = True          # 명확화 단계 활성화
```

## 🤖 AutoGen 에이전트 상세 설명

### 🤔 ClarificationAgent (extends `AssistantAgent`)
```python
class ClarificationAgent(AssistantAgent):
    def __init__(self, model_client):
        super().__init__(
            name="ClarificationAgent",
            model_client=model_client,
            system_message="명확화 전문가 프롬프트..."
        )
```
- **AutoGen 기능**: 기본 `AssistantAgent` 패턴, 구조화된 응답 생성
- **역할**: 사용자 요청의 모호한 부분을 식별하고 명확화 질문 생성
- **기능**: 약어, 전문용어, 불명확한 표현에 대한 추가 정보 요청
- **출력**: JSON 형태의 명확화 요청 (`ClarificationRequest` 모델)

### 📋 ResearchBriefAgent (extends `AssistantAgent`)
```python
class ResearchBriefAgent(AssistantAgent):
    def __init__(self, model_client):
        super().__init__(
            name="ResearchBriefAgent", 
            model_client=model_client,
            system_message="연구 브리프 생성 프롬프트..."
        )
```
- **AutoGen 기능**: `AssistantAgent`의 시스템 메시지 활용
- **역할**: 사용자 요청을 상세하고 구체적인 연구 계획으로 변환
- **기능**: 연구 범위 정의, 핵심 질문 도출, 연구 방향 설정
- **출력**: 구조화된 연구 브리프 (`ResearchBrief` 모델)

### 🎯 SupervisorAgent (extends `AssistantAgent` + `FunctionTool`)
```python
class SupervisorAgent(AssistantAgent):
    def __init__(self, model_client):
        tools = [
            FunctionTool(func=self._conduct_research, name="conduct_research"),
            FunctionTool(func=self._research_complete, name="research_complete")
        ]
        super().__init__(
            name="SupervisorAgent",
            model_client=model_client,
            tools=tools,
            system_message=SUPERVISOR_INSTRUCTIONS
        )
```
- **AutoGen 기능**: `FunctionTool` 통합, `parallel_tool_calls` 지원
- **역할**: 전체 연구 프로세스 오케스트레이션 및 AI 기반 완료 판정
- **도구**: 
  - `conduct_research`: 연구 작업 할당
  - `research_complete`: 연구 완료 신호
- **핵심**: 하드코딩된 임계값 대신 LLM이 연구 충분성을 지능적으로 판단

### 🔍 ResearcherAgent (extends `AssistantAgent` + Multiple `FunctionTool`)
```python
class ResearcherAgent(AssistantAgent):
    def __init__(self, name, model_client, search_tool, mcp_tools):
        tools = [
            FunctionTool(func=self._wrapped_web_search, name="web_search"),
            FunctionTool(func=self._research_complete, name="research_complete")
        ]
        if mcp_tools:
            tools.extend(mcp_tools)
            
        super().__init__(
            name=name,
            model_client=model_client,
            tools=tools
        )
```
- **AutoGen 기능**: 다중 `FunctionTool` 통합, 검색 활동 추적
- **역할**: 할당된 주제에 대한 실제 연구 수행
- **도구**: 
  - `web_search`: Tavily/DuckDuckGo 검색
  - `research_complete`: 연구 완료 (검증 포함)
  - MCP 도구들 (선택적)
- **검증**: 최소 2회 이상 검색 수행 후 연구 완료 가능

### 📊 CompressionAgent (extends `AssistantAgent`)
```python
class CompressionAgent(AssistantAgent):
    def __init__(self, model_client):
        super().__init__(
            name="CompressionAgent",
            model_client=model_client,
            system_message=compress_research_system_prompt
        )
```
- **AutoGen 기능**: 전문화된 시스템 프롬프트로 결과 압축
- **역할**: 여러 연구 결과를 정리하고 중복 제거
- **기능**: 
  - 연구 결과 통합 및 정리
  - 중복 정보 제거
  - 소스 정보 보존
- **출력**: 압축된 연구 결과 (`CompressedResearch` 모델)

### 📄 ReportWriterAgent (extends `AssistantAgent`)
```python
class ReportWriterAgent(AssistantAgent):
    def __init__(self, model_client):
        super().__init__(
            name="ReportWriterAgent",
            model_client=model_client,
            system_message="전문 보고서 작성 프롬프트..."
        )
```
- **AutoGen 기능**: 마크다운 형식의 구조화된 출력 생성
- **역할**: 최종 마크다운 보고서 생성
- **기능**:
  - 구조화된 보고서 작성
  - 적절한 인용 및 출처 표기
  - 마크다운 형식 적용
- **출력**: 완성된 연구 보고서 (`FinalReport` 모델)

## 🔄 워크플로우

### 1. 초기화 단계
```python
state = create_initial_state(workflow_id, user_request)
```

### 2. 명확화 단계 (선택적)
```python
if config.allow_clarification:
    clarification_result = await clarify_request(state)
```

### 3. 연구 계획 단계
```python
research_brief = await create_research_brief(state)
```

### 4. 연구 수행 단계
```python
# 감독자가 연구 작업 계획
tasks = await supervisor_plan_research(state)

# 병렬 연구 수행
research_results = await conduct_parallel_research(tasks)

# AI 기반 완료 판정
is_sufficient = await is_research_sufficient(state)
```

### 5. 결과 압축 단계
```python
compressed_research = await compress_research(state)
```

### 6. 보고서 생성 단계
```python
final_report = await generate_report(state)
```

## 📋 예제

### 예제 1: 기술 동향 연구

```python
request = """
최신 AI 기술 동향을 조사해주세요. 특히 다음 분야에 초점을 맞춰주세요:
1. 대규모 언어 모델 (LLM)
2. 컴퓨터 비전
3. 강화학습
4. 실제 산업 적용 사례
"""

report = await run_deep_research(request)
```

### 예제 2: 시장 분석

```python
request = """
전기차 시장의 현재 상황과 향후 5년 전망을 분석해주세요.
주요 제조업체, 기술 혁신, 정부 정책, 소비자 트렌드를 포함해주세요.
"""

report = await run_deep_research(request)
```

### 예제 3: 학술 연구

```python
request = """
기후변화가 해양 생태계에 미치는 영향에 대한 최신 연구 동향을 조사해주세요.
특히 산호초 생태계와 해양 생물 다양성에 초점을 맞춰주세요.
"""

report = await run_deep_research(request)
```

## 🔧 문제해결

### 일반적인 문제

#### 1. API 키 오류
```bash
# 환경변수 확인
echo $AZURE_OPENAI_API_KEY
echo $TAVILY_API_KEY

# .env 파일 확인
cat .env
```

#### 2. 토큰 한계 초과
- 더 작은 모델 사용 고려 (gpt-4o-mini)
- `max_concurrent_research_units` 감소
- `max_react_tool_calls` 감소

#### 3. 검색 결과 부족
```python
# 여러 검색 API 동시 사용
SEARCH_APIS=tavily,duckduckgo

# 검색 API 키 확인
TAVILY_API_KEY=your-valid-key
```

#### 4. 연구 완료 판정 문제
- SupervisorAgent의 지능적 판정 시스템이 작동
- 필요시 `max_researcher_iterations` 조정

#### 5. parallel_tool_calls 오류
**오류**: `'parallel_tool_calls' is only allowed when 'tools' are specified`
- **자동 해결**: 시스템이 자동으로 깨끗한 모델 클라이언트를 재생성
- **원인**: 도구 없는 작업에서 `parallel_tool_calls=True` 설정 충돌
- **해결책**: 내장된 오류 처리가 자동으로 처리함 (사용자 개입 불필요)

#### 6. Azure OpenAI 콘텐츠 필터 오류
**오류**: Azure OpenAI의 콘텐츠 필터가 연구 내용을 차단하는 경우
```python
# 자동 처리됨 - 다음과 같은 기능들이 내장:
- 콘텐츠 필터 오류 자동 감지
- 대체 검색어로 자동 재시도
- 안전한 연구 내용으로 자동 변환
- 연구 진행 중단 없이 지속적 처리
```

#### 7. AI 기반 작업 분해 실패
**증상**: 연구 작업이 단순한 형태로만 생성됨
- **Fallback 시스템**: AI 실패 시 체계적인 기본 분해 방식으로 자동 전환
- **로그 확인**: `WARNING: AI 기반 작업 생성 실패, 기본 방식 사용` 메시지
- **성능 영향**: 기본 방식도 충분히 효과적이므로 품질 저하 없음
- 로그 확인으로 판정 근거 파악

### 로깅 설정

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### 성능 모니터링

```python
# 연구 메타데이터 확인
print(f"연구 소요 시간: {report.metadata['duration']}초")
print(f"총 반복 횟수: {report.metadata['total_iterations']}")
print(f"사용된 연구자: {report.metadata['researchers_used']}")
print(f"고유 소스: {report.metadata['unique_sources']}")
```

## 🏷️ 주요 클래스 및 함수

### 메인 함수
```python
from autogen_open_deep_research import run_deep_research

# 간단한 연구 실행
report = await run_deep_research("연구 주제")
```

### 설정 클래스
```python
from autogen_open_deep_research import Configuration, load_configuration

# 설정 로드
config = load_configuration()

# 커스텀 설정
config = Configuration(
    max_concurrent_research_units=3,
    allow_clarification=False
)
```

### 상태 관리
```python
from autogen_open_deep_research import (
    ResearchState, 
    create_initial_state,
    StateManager
)

# 상태 생성 및 관리
state = create_initial_state("workflow_id", "user_request")
state_manager = StateManager()
state_manager.save_state(state)
```

## 📁 프로젝트 구조

```
autogen_open_deep_research/
├── __init__.py              # 패키지 초기화 및 메인 익스포트
├── configuration.py         # 설정 관리 (모델, API, 파라미터)
├── deep_researcher.py       # 메인 연구 시스템 및 에이전트들
├── state.py                # 상태 관리 클래스들
├── tools.py                # 검색 및 외부 도구 통합
├── utils.py                # 유틸리티 함수들
├── prompts.py              # 에이전트 프롬프트 정의
├── test_examples.py        # 테스트 예제들
├── workflow_test.py        # 워크플로우 테스트
├── env.example             # 환경변수 예시
└── README.md              # 이 파일
```

## 🔮 AutoGen 기반 향후 계획

### 🚀 **AutoGen 고급 기능 확장**
- **`GraphFlow` 통합**: 복잡한 다단계 워크플로우 라우팅 구현
- **`SelectorGroupChat` 완전 구현**: 지능적 에이전트 선택 시스템
- **`RoundRobinGroupChat`**: 순환 에이전트 대화 패턴 적용
- **`UserProxyAgent` 통합**: 사용자 승인 단계 추가

### 🔧 **추가 AutoGen 기능 활용**
- **`HandoffMessage`**: 에이전트 간 명시적 작업 전달
- **Enhanced Termination Conditions**: 더 정교한 종료 조건 시스템
- **Distributed Runtime**: 분산 에이전트 실행 환경
- **Message Streaming**: 실시간 메시지 스트리밍

### 🌐 **외부 통합 확장**
- **MCP (Model Context Protocol)**: 외부 데이터베이스 및 파일 시스템 연동
- **더 많은 FunctionTool**: 전문 도구 생태계 구축
- **결과 캐싱**: AutoGen 메시지 레벨에서 중복 방지
- **다국어 지원**: 다양한 언어로 연구 수행

### 👥 **협업 및 확장성**
- **Multi-tenant 지원**: 여러 사용자 동시 연구
- **Agent Team Templates**: 재사용 가능한 에이전트 팀 구성
- **Custom Agent Types**: 도메인별 전문 에이전트 개발

### 📊 **모니터링 및 최적화**
- **AutoGen Telemetry**: OpenTelemetry 기반 성능 모니터링
- **Agent Performance Analytics**: 에이전트별 성능 분석
- **Cost Optimization**: 모델 호출 비용 최적화

## 🤝 기여하기

1. 이 저장소를 포크합니다
2. 기능 브랜치를 생성합니다 (`git checkout -b feature/AmazingFeature`)
3. 변경사항을 커밋합니다 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치에 푸시합니다 (`git push origin feature/AmazingFeature`)
5. Pull Request를 생성합니다

## 📄 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 👨‍💻 작성자

**김재현 (Kim Jaehyun)**
- 📧 Email: hyun030508@gmail.com
- 🏢 AutoGen Deep Research System 개발자

## 🙏 감사의 말

### 🎯 **핵심 기술 스택**
- **[Microsoft AutoGen](https://github.com/microsoft/autogen)** 🤖: 본 시스템의 핵심 다중 에이전트 프레임워크
  - `AssistantAgent`, `FunctionTool`, Model Clients 등 핵심 컴포넌트 활용
  - 향후 `GraphFlow`, `SelectorGroupChat` 등 고급 기능 확장 예정

### 🔍 **검색 및 AI 서비스**
- **[Tavily Search API](https://tavily.com/)** 🔍: 고품질 웹 검색 서비스
- **[OpenAI](https://openai.com/)** 🧠: GPT-4o, GPT-4o-mini 모델 제공
- **[Azure OpenAI Service](https://azure.microsoft.com/en-us/products/ai-services/openai-service)** ☁️: 엔터프라이즈급 AI 서비스


### 🛠️ **개발 도구 및 라이브러리**
- **[Pydantic](https://pydantic.dev/)**: 데이터 검증 및 모델링
- **[asyncio](https://docs.python.org/3/library/asyncio.html)**: 비동기 프로그래밍 지원
- **Python Ecosystem**: 안정적인 개발 환경 제공

---

## 📞 지원 및 문의

문제가 발생하거나 질문이 있으시면 GitHub Issues를 통해 문의해주세요.

**Happy Researching! 🔬✨**
