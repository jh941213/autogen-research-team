import asyncio
import os
from datetime import datetime
from pathlib import Path

from deep_researcher import DeepResearchTeam
from configuration import Configuration


async def simple_test():
    "테스트"
    
    print("🔍 테스트")
    print("=" * 40)
    
    # 쿼리 입력
    query = "메타가 오픈ai 연구원을 빼돌리는 이유"
    print(f"검색어: {query}")
    
    print(f"\n🚀 검색 시작: '{query}'")
    print("⏳ 처리 중...")
    
    try:
        # 설정 및 연구 시스템 초기화
        config = Configuration.from_env()
        researcher = DeepResearchTeam(config)
        
        # 연구 실행
        start_time = datetime.now()
        result = await researcher.process_research_request(query)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ 완료! ({duration:.1f}초)")
        
        # 결과 출력
        if hasattr(result, 'title'):
            print(f"📄 제목: {result.title}")
        
        # MD 파일로 저장
        report_dir = Path("report")
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simple_test_{timestamp}.md"
        filepath = report_dir / filename
        
        # MD 내용 생성
        md_content = []
        md_content.append(f"# {result.title if hasattr(result, 'title') else query}")
        md_content.append("")
        md_content.append(f"**검색어**: {query}")
        md_content.append(f"**생성 시간**: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        md_content.append(f"**소요 시간**: {duration:.1f}초")
        md_content.append("")
        md_content.append("## 연구 결과")
        md_content.append("")
        
        if hasattr(result, 'executive_summary'):
            md_content.append(result.executive_summary)
        else:
            md_content.append(str(result))
        
        md_content.append("")
        
        # 소스 정보 추가
        if hasattr(result, 'sources') and result.sources:
            md_content.append("## 참고 자료")
            md_content.append("")
            for i, source in enumerate(result.sources, 1):
                md_content.append(f"{i}. {source}")
            md_content.append("")
        
        # 파일 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_content))
        
        print(f"💾 결과 저장: {filepath}")
        print("\n🎉 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")


if __name__ == "__main__":
    asyncio.run(simple_test())