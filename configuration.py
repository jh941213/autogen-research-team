"""
AutoGen Deep Research 시스템 설정 모듈

사용 예시:

1. OpenAI 사용 시 환경변수 설정:
   export OPENAI_API_KEY="your-openai-api-key"
   export MODEL_PROVIDER="openai"
   export OPENAI_RESEARCH_MODEL="gpt-4o"
   export OPENAI_SUMMARIZATION_MODEL="gpt-4o-mini"
   export OPENAI_COMPRESSION_MODEL="gpt-4o-mini"
   export OPENAI_FINAL_REPORT_MODEL="gpt-4o"

2. Azure OpenAI 사용 시 환경변수 설정:
   export AZURE_OPENAI_API_KEY="your-azure-key"
   export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
   export AZURE_OPENAI_DEPLOYMENT_NAME="your-deployment"
   export MODEL_PROVIDER="azure_openai"

3. 검색 API 설정:
   export TAVILY_API_KEY="your-tavily-key"
   export SEARCH_APIS="tavily,duckduckgo"  # 여러 API 동시 사용 가능

4. 코드에서 사용:
   from configuration import load_configuration
   config = load_configuration()
   clients = config.get_model_clients()
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
import os
from dotenv import load_dotenv
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient, OpenAIChatCompletionClient
from autogen_core.models import ModelInfo

load_dotenv()


# Search API Types - 연구 에이전트가 사용할 수 있는 웹 검색 API
class SearchAPI(Enum):
    """연구 에이전트가 사용할 수 있는 검색 API들"""
    TAVILY = "tavily"           # Tavily Search API (기본값)
    DUCKDUCKGO = "duckduckgo"   # DuckDuckGo Search API
    NONE = "none"               # 검색 비활성화
    

# Model Provider Types - 사용 가능한 LLM 제공자
class ModelProvider(Enum):
    """지원되는 모델 제공자들"""
    AZURE_OPENAI = "azure_openai"
    OPENAI = "openai"
    LOCAL = "local"  # 로컬/커스텀 모델용


# MCP (Model Context Protocol) 서버 설정
class MCPConfig(BaseModel):
    """MCP 서버 통합을 위한 설정"""
    url: Optional[str] = Field(
        default=None,
        description="MCP 서버 URL 주소"
    )
    tools: Optional[List[str]] = Field(
        default=None,
        description="LLM이 사용할 수 있는 도구 목록 (예: 파일 읽기, 데이터베이스 쿼리)"
    )
    auth_required: bool = Field(
        default=False,
        description="MCP 서버 접근에 인증이 필요한지 여부"
    )


# Model Configuration - 개별 모델 설정
class ModelConfig(BaseModel):
    """개별 모델 인스턴스를 위한 설정"""
    provider: ModelProvider = Field(
        default=ModelProvider.AZURE_OPENAI,
        description="모델 제공자 유형"
    )
    model_name: str = Field(
        description="모델 이름 (예: gpt-4o, gpt-4o-mini)"
    )
    deployment_name: Optional[str] = Field(
        default=None,
        description="Azure OpenAI 배포 이름"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API 키 (환경변수 사용 가능)"
    )
    azure_endpoint: Optional[str] = Field(
        default=None,
        description="Azure OpenAI 엔드포인트 URL"
    )
    api_version: Optional[str] = Field(
        default="2024-08-01-preview",
        description="Azure OpenAI API 버전"
    )
    base_url: Optional[str] = Field(
        default=None,
        description="로컬/커스텀 모델용 베이스 URL"
    )
    max_tokens: int = Field(
        default=4096,
        description="최대 출력 토큰 수"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="응답 무작위성을 위한 모델 온도"
    )
    parallel_tool_calls: bool = Field(
        default=False,
        description="병렬 도구 실행 활성화"
    )
    
    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v: Optional[str], info) -> Optional[str]:
        """제공되지 않은 경우 환경변수에서 API 키 가져오기"""
        if v:
            return v
            
        provider = info.data.get('provider')
        if provider == ModelProvider.AZURE_OPENAI:
            return os.getenv("AZURE_OPENAI_API_KEY")
        elif provider == ModelProvider.OPENAI:
            return os.getenv("OPENAI_API_KEY")
        return None
    
    @field_validator('azure_endpoint')
    @classmethod
    def validate_azure_endpoint(cls, v: Optional[str]) -> Optional[str]:
        """제공되지 않은 경우 환경변수에서 Azure 엔드포인트 가져오기"""
        return v or os.getenv("AZURE_OPENAI_ENDPOINT")
    
    def to_client(self) -> Union[AzureOpenAIChatCompletionClient, OpenAIChatCompletionClient]:
        """설정을 AutoGen 모델 클라이언트로 변환"""
        if self.provider == ModelProvider.AZURE_OPENAI:
            # 환경변수를 폴백으로 사용
            deployment_name = self.deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            azure_endpoint = self.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = self.api_key or os.getenv("AZURE_OPENAI_API_KEY")
            
            if not all([deployment_name, azure_endpoint, api_key]):
                raise ValueError("Azure OpenAI requires deployment_name, azure_endpoint, and api_key")
            
            # 클라이언트 인수 준비
            client_kwargs = {
                "azure_deployment": deployment_name,
                "model": self.model_name,
                "api_version": self.api_version,
                "azure_endpoint": azure_endpoint,
                "api_key": api_key,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
            
            # True인 경우에만 parallel_tool_calls 추가 (도구가 제공될 예정)
            if self.parallel_tool_calls:
                client_kwargs["parallel_tool_calls"] = self.parallel_tool_calls
                
            return AzureOpenAIChatCompletionClient(**client_kwargs)
            
        elif self.provider == ModelProvider.OPENAI:
            # 클라이언트 인수 준비
            client_kwargs = {
                "model": self.model_name,
                "api_key": self.api_key,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
            
            # True인 경우에만 parallel_tool_calls 추가 (도구가 제공될 예정)
            if self.parallel_tool_calls:
                client_kwargs["parallel_tool_calls"] = self.parallel_tool_calls
                
            return OpenAIChatCompletionClient(**client_kwargs)
            
        elif self.provider == ModelProvider.LOCAL:
            if not self.base_url:
                raise ValueError("Local models require base_url")
                
            return OpenAIChatCompletionClient(
                model=self.model_name,
                base_url=self.base_url,
                api_key=self.api_key or "not-required",
                model_info=ModelInfo(
                    vision=False,
                    function_calling=True,
                    json_output=True,
                    family="unknown",
                    structured_output=True
                ),
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

# 메인 설정 클래스
class Configuration(BaseModel):
    """
    AutoGen Deep Research 시스템을 위한 메인 설정 클래스.
    모델, 검색 API, 연구 파라미터를 포함한 모든 시스템 구성요소를 관리합니다.
    """
    max_structured_output_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="구조화된 출력 생성을 위한 최대 재시도 횟수"
    )
    
    allow_clarification: bool = Field(
        default=True,
        description="연구 전 에이전트가 명확화 질문을 할 수 있도록 허용"
    )
    
    max_concurrent_research_units: int = Field(
        default=5,
        ge=1,
        le=20,
        description="최대 동시 연구 단위 수 (속도 제한에 주의)"
    )
    
# 리서처 설정 값
    
    search_apis: List[SearchAPI] = Field(
        default_factory=lambda: [SearchAPI.TAVILY],
        description="연구에 사용할 검색 API 목록 (동시에 여러 개 사용 가능)"
    )
    
    search_api_key: Optional[str] = Field(
        default=None,
        description="검색 서비스용 API 키"
    )
    
    max_researcher_iterations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="연구 관리자의 최대 반복 횟수"
    )
    
    max_react_tool_calls: int = Field(
        default=5,
        ge=1,
        le=30,
        description="연구자 단계당 최대 도구 호출 횟수"
    )
    
# 모델 설정 값
    
    # Summarization Model - 경량 모델로 검색 결과 요약
    summarization_model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            provider=ModelProvider.AZURE_OPENAI,
            model_name="gpt-4o-mini",
            deployment_name=os.getenv("AZURE_OPENAI_SUMMARIZATION_DEPLOYMENT", "gpt-4o-mini"),
            max_tokens=8192,
            temperature=0.5
        ),
        description="검색 결과 요약용 모델"
    )
    
    # Research Model - 메인 연구 수행 모델
    research_model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            provider=ModelProvider.AZURE_OPENAI,
            model_name="gpt-4o",
            deployment_name=os.getenv("AZURE_OPENAI_RESEARCH_DEPLOYMENT", "gpt-4o"),
            max_tokens=10000,
            temperature=0.7
        ),
        description="연구 수행용 메인 모델"
    )
    
    # Compression Model - 연구 결과 압축 및 정리
    compression_model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            provider=ModelProvider.AZURE_OPENAI,
            model_name="gpt-4o-mini",
            deployment_name=os.getenv("AZURE_OPENAI_COMPRESSION_DEPLOYMENT", "gpt-4o-mini"),
            max_tokens=8192,
            temperature=0.5
        ),
        description="연구 결과 압축용 모델"
    )
    
    # Final Report Model - 최종 보고서 작성
    final_report_model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            provider=ModelProvider.AZURE_OPENAI,
            model_name="gpt-4o",
            deployment_name=os.getenv("AZURE_OPENAI_REPORT_DEPLOYMENT", "gpt-4o"),
            max_tokens=10000,
            temperature=0.7
        ),
        description="최종 보고서 작성용 모델"
    )
    
    # ============================================
    # MCP Server Configuration
    # ============================================
    
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        description="외부 데이터 접근을 위한 MCP 서버 설정"
    )
    
    mcp_prompt: Optional[str] = Field(
        default=None,
        description="MCP 도구 사용을 위한 추가 지침"
    )
    
    # ============================================
    # AutoGen-specific Configuration
    # ============================================
    
    runtime_type: str = Field(
        default="single_threaded",
        description="AutoGen 런타임 유형 (single_threaded 또는 distributed)"
    )
    
    enable_telemetry: bool = Field(
        default=False,
        description="OpenTelemetry 계측 활성화"
    )
    
    cache_enabled: bool = Field(
        default=True,
        description="모델 호출에 대한 응답 캐싱 활성화"
    )
    
    cache_directory: str = Field(
        default=".cache/autogen",
        description="모델 응답 캐싱용 디렉토리"
    )
    
    # ============================================
    # Helper Methods
    # ============================================
    
    @model_validator(mode='after')
    def set_search_api_key(self) -> 'Configuration':
        """제공되지 않은 경우 환경변수에서 검색 API 키 설정"""
        if not self.search_api_key:
            # Tavily가 목록에 있으면 Tavily 키 설정
            if SearchAPI.TAVILY in self.search_apis:
                self.search_api_key = os.getenv("TAVILY_API_KEY")
            # DuckDuckGo는 API 키가 필요하지 않으므로 설정하지 않음
        return self
    
    def get_model_clients(self) -> Dict[str, Any]:
        """설정된 모든 모델 클라이언트 가져오기"""
        return {
            "summarization": self.summarization_model.to_client(),
            "research": self.research_model.to_client(),
            "compression": self.compression_model.to_client(),
            "final_report": self.final_report_model.to_client()
        }
    
    def get_model_client_with_tools(self, model_type: str) -> Any:
        """
        도구 사용을 위해 설정된 모델 클라이언트 가져오기 (parallel_tool_calls 활성화됨)
        
        Args:
            model_type: 모델 유형 ("research", "summarization" 등)
            
        Returns:
            parallel_tool_calls가 활성화된 모델 클라이언트
        """
        model_config = getattr(self, f"{model_type}_model")
        
        # parallel_tool_calls가 활성화된 모델 설정의 복사본 생성
        tool_config = model_config.model_copy()
        tool_config.parallel_tool_calls = True
        
        return tool_config.to_client()
    
    @classmethod
    def from_env(cls) -> "Configuration":
        """환경변수에서 설정 생성"""
        # 기본값으로 환경변수에서 로드
        config_dict = {}
        
        # 검색 API 설정 확인 - 다중 API 지원
        search_apis_env = os.getenv("SEARCH_APIS")  # "tavily,duckduckgo" 형태
        if search_apis_env:
            try:
                api_names = [name.strip().lower() for name in search_apis_env.split(",")]
                search_apis = []
                for api_name in api_names:
                    if api_name == "tavily":
                        search_apis.append(SearchAPI.TAVILY)
                    elif api_name == "duckduckgo":
                        search_apis.append(SearchAPI.DUCKDUCKGO)
                    elif api_name == "none":
                        search_apis.append(SearchAPI.NONE)
                config_dict["search_apis"] = search_apis if search_apis else [SearchAPI.TAVILY]
            except Exception:
                # 파싱 실패시 기본값 사용
                config_dict["search_apis"] = [SearchAPI.TAVILY] if os.getenv("TAVILY_API_KEY") else [SearchAPI.DUCKDUCKGO]
        else:
            # 환경변수가 없으면 기본값 설정
            available_apis = []
            if os.getenv("TAVILY_API_KEY"):
                available_apis.append(SearchAPI.TAVILY)
            available_apis.append(SearchAPI.DUCKDUCKGO)  # DuckDuckGo는 항상 사용 가능
            config_dict["search_apis"] = available_apis
            
        # 사용 가능한 경우 검색 API 키 설정
        if os.getenv("TAVILY_API_KEY"):
            config_dict["search_api_key"] = os.getenv("TAVILY_API_KEY")
        
        # 모델 제공자 선호도 확인
        default_provider = ModelProvider.AZURE_OPENAI
        if os.getenv("MODEL_PROVIDER"):
            default_provider = ModelProvider(os.getenv("MODEL_PROVIDER"))
        
        # 제공자에 따른 모델 설정
        if default_provider == ModelProvider.AZURE_OPENAI:
            # Azure OpenAI 설정
            if os.getenv("AZURE_OPENAI_ENDPOINT"):
                base_config = {
                    "provider": ModelProvider.AZURE_OPENAI,
                    "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
                    "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
                    "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
                }
                
                # 특정 배포로 각 모델 설정
                default_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
                
                config_dict["summarization_model"] = ModelConfig(
                    **base_config,
                    model_name=os.getenv("SUMMARIZATION_MODEL", "gpt-4o-mini"),
                    deployment_name=os.getenv("AZURE_OPENAI_SUMMARIZATION_DEPLOYMENT", default_deployment)
                )
                
                config_dict["research_model"] = ModelConfig(
                    **base_config,
                    model_name=os.getenv("RESEARCH_MODEL", "gpt-4o"),
                    deployment_name=os.getenv("AZURE_OPENAI_RESEARCH_DEPLOYMENT", default_deployment)
                )
                
                config_dict["compression_model"] = ModelConfig(
                    **base_config,
                    model_name=os.getenv("COMPRESSION_MODEL", "gpt-4o-mini"),
                    deployment_name=os.getenv("AZURE_OPENAI_COMPRESSION_DEPLOYMENT", default_deployment)
                )
                
                config_dict["final_report_model"] = ModelConfig(
                    **base_config,
                    model_name=os.getenv("FINAL_REPORT_MODEL", "gpt-4o"),
                    deployment_name=os.getenv("AZURE_OPENAI_FINAL_REPORT_DEPLOYMENT", default_deployment)
                )
                    
        elif default_provider == ModelProvider.OPENAI:
            # OpenAI 설정
            base_config = {
                "provider": ModelProvider.OPENAI,
                "api_key": os.getenv("OPENAI_API_KEY")
            }
            
            config_dict["summarization_model"] = ModelConfig(
                **base_config,
                model_name=os.getenv("OPENAI_SUMMARIZATION_MODEL", "gpt-4o-mini")
            )
            
            config_dict["research_model"] = ModelConfig(
                **base_config,
                model_name=os.getenv("OPENAI_RESEARCH_MODEL", "gpt-4o")
            )
            
            config_dict["compression_model"] = ModelConfig(
                **base_config,
                model_name=os.getenv("OPENAI_COMPRESSION_MODEL", "gpt-4o-mini")
            )
            
            config_dict["final_report_model"] = ModelConfig(
                **base_config,
                model_name=os.getenv("OPENAI_FINAL_REPORT_MODEL", "gpt-4o")
            )
        
        # 기타 설정
        if os.getenv("MAX_CONCURRENT_RESEARCH_UNITS"):
            config_dict["max_concurrent_research_units"] = int(os.getenv("MAX_CONCURRENT_RESEARCH_UNITS"))
        
        if os.getenv("ENABLE_TELEMETRY"):
            config_dict["enable_telemetry"] = os.getenv("ENABLE_TELEMETRY").lower() == "true"
        
        config = cls(**config_dict)
        
        # 열거형 타입이 올바른지 확인하기 위한 후처리
        config = config.set_search_api_key()
        
        # search_apis가 올바른 열거형 타입인지 확인 (Pydantic use_enum_values=True로 인한 문자열 변환)
        if hasattr(config, 'search_apis') and config.search_apis:
            corrected_apis = []
            for api in config.search_apis:
                if isinstance(api, str):
                    if api.lower() == "tavily":
                        corrected_apis.append(SearchAPI.TAVILY)
                    elif api.lower() == "duckduckgo":
                        corrected_apis.append(SearchAPI.DUCKDUCKGO)
                    elif api.lower() == "none":
                        corrected_apis.append(SearchAPI.NONE)
                else:
                    corrected_apis.append(api)
            config.search_apis = corrected_apis
                
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """직렬화를 위해 설정을 딕셔너리로 변환"""
        return self.model_dump(exclude_none=True)
    
    class Config:
        """Pydantic 설정"""
        arbitrary_types_allowed = True
        use_enum_values = True


# 설정 로딩을 위한 편의 함수
def load_configuration(config_path: Optional[str] = None) -> Configuration:
    """
    파일 또는 환경변수에서 설정 로드
    
    Args:
        config_path: 설정 파일 경로 (JSON/YAML) - 선택사항
        
    Returns:
        Configuration 인스턴스
    """
    if config_path and os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return Configuration(**config_data)
    else:
        return Configuration.from_env()