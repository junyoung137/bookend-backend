from typing import Dict, Any, Optional
import logging
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# =========================================================
# Context Type Enums
# =========================================================

class TimeOfDay(str, Enum):
    """시간대 구분 (실제 사용 패턴 기반)"""
    DAWN = "dawn"                    # 0-6   (33.8% 트래픽 - 고요한 창작 모드)
    MORNING = "morning"              # 6-12  (41.6% 트래픽 - 집중형)
    AFTERNOON = "afternoon"          # 12-18 (17.5% 트래픽 - 확장형)
    EVENING = "evening"              # 18-22 (7.1% 트래픽 - 감성형)
    NIGHT = "night"                  # 22-24


class DayType(str, Enum):
    WEEKDAY = "weekday"
    WEEKEND = "weekend"


class DeviceType(str, Enum):
    """Desktop과 Mac만 지원"""
    DESKTOP = "desktop"
    MAC = "mac"
    TABLET = "tablet"
    UNKNOWN = "unknown"


class ActivityLevel(str, Enum):
    """사용자 활동 레벨"""
    NEW = "new"           # 신규 (0회)
    LOW = "low"           # 저활동 (1-4회)
    MEDIUM = "medium"     # 중활동 (5-14회)
    HIGH = "high"         # 고활동 (15회+)
    POWER = "power"       # 파워유저 (50회+)


class EngagementLevel(str, Enum):
    """사용자 몰입도"""
    EXPLORING = "exploring"   # 탐색 중 (< 2분)
    ACTIVE = "active"         # 활성 (2-10분)
    ENGAGED = "engaged"       # 몰입 (10-30분)
    DEEP_WORK = "deep_work"   # 깊은 작업 (30분+)


class RecommendationTone(str, Enum):
    """시간대별 추천 톤 (Temporal Flow 지원)"""
    FOCUSED = "focused"           # 집중형 - 간결하게 다시쓰기
    CREATIVE = "creative"         # 창작형 - 고요한 창작 모드
    EXPANSIVE = "expansive"       # 확장형 - 아이디어 확장
    EMOTIONAL = "emotional"       # 감성형 - 시적 스타일
    NEUTRAL = "neutral"           # 중립


# =========================================================
# Context Feature Extractor
# =========================================================

class ContextFeatureExtractor:
    """
    실시간 추천을 위한 컨텍스트 피처 추출기
    
    지원 기능:
    - Temporal Flow: 시간대별 추천 톤 매핑
    - Echo Feedback: 사용자 선호도 추적
    - Ambient Recommendation: 환경 기반 배치
    - Context Echo: 문맥 기반 제안
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    # =========================================================
    # Temporal Context (시간 컨텍스트)
    # =========================================================

    def extract_temporal_context(
        self, 
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        시간 기반 컨텍스트 추출
        
        실제 통계 반영:
        - 새벽 (00-06): 33.8% 트래픽
        - 오전 (06-12): 41.6% 트래픽  
        - 오후 (12-18): 17.5% 트래픽
        - 저녁 (18-24): 7.1% 트래픽
        """
        try:
            timestamp = timestamp or datetime.now()
            hour = timestamp.hour
            day_of_week = timestamp.weekday()

            time_of_day = self._classify_time_of_day(hour)
            day_type = DayType.WEEKEND if day_of_week >= 5 else DayType.WEEKDAY
            is_business_hours = self._is_business_hours(hour, day_of_week)
            
            # Temporal Flow: 시간대별 추천 톤 매핑
            recommendation_tone = self._get_recommendation_tone(hour)
            
            # 피크 시간대 여부 (07-08시)
            is_peak_hours = hour in [7, 8]

            return {
                'hour': hour,
                'day_of_week': day_of_week,
                'time_of_day': time_of_day.value,
                'day_type': day_type.value,
                'is_business_hours': is_business_hours,
                'is_weekend': day_of_week >= 5,
                'is_peak_hours': is_peak_hours,
                'recommendation_tone': recommendation_tone.value,
                'timestamp': timestamp.isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Failed to extract temporal context: {e}", exc_info=True)
            return self._get_default_temporal_context()

    def _classify_time_of_day(self, hour: int) -> TimeOfDay:
        """실제 사용 패턴 기반 시간대 분류"""
        if 0 <= hour < 6:
            return TimeOfDay.DAWN      # 33.8% 트래픽
        elif 6 <= hour < 12:
            return TimeOfDay.MORNING   # 41.6% 트래픽 (피크: 07시, 08시)
        elif 12 <= hour < 18:
            return TimeOfDay.AFTERNOON # 17.5% 트래픽
        elif 18 <= hour < 22:
            return TimeOfDay.EVENING   # 7.1% 트래픽
        else:
            return TimeOfDay.NIGHT

    def _get_recommendation_tone(self, hour: int) -> RecommendationTone:
        """
        Temporal Flow: 시간대별 추천 톤
        
        | 시간대 | 추천 톤 | 행동 제안 |
        |--------|---------|-----------|
        | 새벽   | 창작형  | 고요한 창작 모드 |
        | 오전   | 집중형  | 간결하게 다시쓰기 |
        | 오후   | 확장형  | 아이디어 확장 |
        | 저녁   | 감성형  | 시적 스타일 제안 |
        """
        if 0 <= hour < 6:
            return RecommendationTone.CREATIVE    # 새벽: 창작형
        elif 6 <= hour < 12:
            return RecommendationTone.FOCUSED     # 오전: 집중형
        elif 12 <= hour < 18:
            return RecommendationTone.EXPANSIVE   # 오후: 확장형
        elif 18 <= hour < 22:
            return RecommendationTone.EMOTIONAL   # 저녁: 감성형
        else:
            return RecommendationTone.NEUTRAL

    def _is_business_hours(self, hour: int, day_of_week: int) -> bool:
        """업무 시간 여부 (평일 09-18시)"""
        return (9 <= hour < 18) and (day_of_week < 5)

    def _get_default_temporal_context(self) -> Dict[str, Any]:
        return {
            'hour': None,
            'day_of_week': None,
            'time_of_day': None,
            'day_type': None,
            'is_business_hours': False,
            'is_weekend': False,
            'is_peak_hours': False,
            'recommendation_tone': RecommendationTone.NEUTRAL.value,
            'timestamp': None,
        }

    # =========================================================
    # Device Context (디바이스 컨텍스트)
    # =========================================================

    def extract_device_context(
        self,
        browser: Optional[str] = None,
        os: Optional[str] = None,
        device_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        디바이스 기반 컨텍스트 추출
        
        실제 통계 반영:
        - Chrome: 59.98%
        - Edge: 2.78%
        - Safari: 1.26%
        - Whale: 1.11%
        """
        try:
            device_type = self._classify_device_type(os)
            browser_family = self._normalize_browser(browser)
            os_family = self._normalize_os(os)
            
            # 주요 브라우저 여부
            is_chrome = browser_family == 'Chrome'
            is_whale = browser_family == 'Whale'
            is_major_browser = is_chrome or is_whale

            return {
                'device_type': device_type.value,
                'browser': browser,
                'browser_family': browser_family,
                'os': os,
                'os_family': os_family,
                'device_id': device_id,
                'is_mac': device_type == DeviceType.MAC,
                'is_desktop': device_type == DeviceType.DESKTOP,
                'is_tablet': device_type == DeviceType.TABLET,
                'is_chrome': is_chrome,
                'is_whale': is_whale,
                'is_major_browser': is_major_browser,
            }

        except Exception as e:
            self.logger.error(f"Failed to extract device context: {e}", exc_info=True)
            return self._get_default_device_context()

    def _classify_device_type(self, os: Optional[str]) -> DeviceType:
        if not os:
            return DeviceType.UNKNOWN

        os_lower = os.lower()

        if any(keyword in os_lower for keyword in ['mac', 'darwin', 'os x']):
            return DeviceType.MAC
        if any(keyword in os_lower for keyword in ['windows', 'linux', 'chrome os']):
            return DeviceType.DESKTOP
        if 'ipad' in os_lower or 'tablet' in os_lower:
            return DeviceType.TABLET
        return DeviceType.DESKTOP  # default fallback

    def _normalize_browser(self, browser: Optional[str]) -> Optional[str]:
        """브라우저 정규화 (실제 통계 기반)"""
        if not browser:
            return None
        browser_lower = browser.lower()
        browser_mapping = {
            'chrome': 'Chrome',
            'firefox': 'Firefox',
            'mozilla': 'Firefox',
            'safari': 'Safari',
            'edge': 'Edge',
            'whale': 'Whale',
            'samsung': 'Samsung Internet',
        }
        for keyword, family in browser_mapping.items():
            if keyword in browser_lower:
                return family
        return 'Other'

    def _normalize_os(self, os: Optional[str]) -> Optional[str]:
        if not os:
            return None
        os_lower = os.lower()
        os_mapping = {
            'windows': 'Windows',
            'mac': 'Mac OS X',
            'darwin': 'Mac OS X',
            'linux': 'Linux',
            'chrome os': 'Chrome OS',
            'android': 'Android',
        }
        for keyword, family in os_mapping.items():
            if keyword in os_lower:
                return family
        return 'Other'

    def _get_default_device_context(self) -> Dict[str, Any]:
        return {
            'device_type': DeviceType.UNKNOWN.value,
            'browser': None,
            'browser_family': None,
            'os': None,
            'os_family': None,
            'device_id': None,
            'is_mac': False,
            'is_desktop': False,
            'is_tablet': False,
            'is_chrome': False,
            'is_whale': False,
            'is_major_browser': False,
        }

    # =========================================================
    # Location Context (위치 컨텍스트)
    # =========================================================

    def extract_location_context(
        self,
        country_code: Optional[str] = None,
        city: Optional[str] = None,
        timezone: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        위치 기반 컨텍스트 추출
        
        실제 통계 반영:
        - 한국 사용자: 76.6% (5,126명)
        - 미국 사용자: 17.8% (1,189명)
        """
        try:
            region = self._classify_region(country_code)
            cc_norm = country_code.upper() if country_code else None
            
            # 한국 사용자 특화
            is_korea = cc_norm == 'KR' if cc_norm else False
            is_seoul_area = is_korea and city and any(
                area in city for area in ['Seoul', 'Gangnam', 'Seocho', 'Gangdong', 'Yongsan']
            )

            return {
                'country_code': country_code,
                'city': city,
                'timezone': timezone,
                'region': region,
                'is_korea': is_korea,
                'is_seoul_area': is_seoul_area,
                'is_asia': region == 'Asia',
                'is_americas': region == 'Americas',
                'is_europe': region == 'Europe',
            }
        except Exception as e:
            self.logger.error(f"Failed to extract location context: {e}", exc_info=True)
            return self._get_default_location_context()

    def _classify_region(self, country_code: Optional[str]) -> Optional[str]:
        if not country_code:
            return None
        cc = country_code.upper()
        region_mapping = {
            'Asia': ['KR', 'JP', 'CN', 'TW', 'SG', 'TH', 'VN', 'IN', 'ID', 'MY', 'PH'],
            'Europe': ['GB', 'DE', 'FR', 'IT', 'ES', 'NL', 'SE', 'NO', 'FI', 'DK', 'PL'],
            'Americas': ['US', 'CA', 'BR', 'MX', 'AR', 'CL', 'CO', 'PE'],
            'Oceania': ['AU', 'NZ'],
        }
        for region, countries in region_mapping.items():
            if cc in countries:
                return region
        return 'Other'

    def _get_default_location_context(self) -> Dict[str, Any]:
        return {
            'country_code': None,
            'city': None,
            'timezone': None,
            'region': None,
            'is_korea': False,
            'is_seoul_area': False,
            'is_asia': False,
            'is_americas': False,
            'is_europe': False,
        }

    # =========================================================
    # Session Context (세션 컨텍스트)
    # =========================================================

    def extract_session_context(
        self,
        events_in_session: int = 0,
        session_duration_minutes: float = 0.0,
        last_event_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        세션 기반 컨텍스트 추출
        
        실제 통계 반영:
        - 파워 유저: 72명 (평균 92.6회/유저)
        - 성장 유저: 116명
        - 신규 유저: 2,010명
        """
        try:
            activity_level = self._classify_activity_level(events_in_session)
            engagement_level = self._classify_engagement_level(session_duration_minutes)
            
            # Echo Feedback: 반복 사용 패턴 감지
            is_repeat_user = events_in_session >= 10
            is_power_user = events_in_session >= 50

            return {
                'events_in_session': events_in_session,
                'session_duration_minutes': session_duration_minutes,
                'last_event_name': last_event_name,
                'activity_level': activity_level.value,
                'engagement_level': engagement_level.value,
                'is_new_session': events_in_session == 0,
                'is_active_session': events_in_session >= 5,
                'is_repeat_user': is_repeat_user,
                'is_power_user': is_power_user,
            }
        except Exception as e:
            self.logger.error(f"Failed to extract session context: {e}", exc_info=True)
            return self._get_default_session_context()

    def _classify_activity_level(self, events_in_session: int) -> ActivityLevel:
        """활동 레벨 분류 (실제 통계 기반)"""
        if events_in_session == 0:
            return ActivityLevel.NEW
        elif events_in_session < 5:
            return ActivityLevel.LOW
        elif events_in_session < 15:
            return ActivityLevel.MEDIUM
        elif events_in_session < 50:
            return ActivityLevel.HIGH
        else:
            return ActivityLevel.POWER  # 파워유저

    def _classify_engagement_level(self, session_duration_minutes: float) -> EngagementLevel:
        """몰입도 분류"""
        if session_duration_minutes < 2:
            return EngagementLevel.EXPLORING
        elif session_duration_minutes < 10:
            return EngagementLevel.ACTIVE
        elif session_duration_minutes < 30:
            return EngagementLevel.ENGAGED
        else:
            return EngagementLevel.DEEP_WORK

    def _get_default_session_context(self) -> Dict[str, Any]:
        return {
            'events_in_session': 0,
            'session_duration_minutes': 0.0,
            'last_event_name': None,
            'activity_level': ActivityLevel.NEW.value,
            'engagement_level': EngagementLevel.EXPLORING.value,
            'is_new_session': True,
            'is_active_session': False,
            'is_repeat_user': False,
            'is_power_user': False,
        }

    # =========================================================
    # User Preference Context (사용자 선호도 컨텍스트)
    # =========================================================

    def extract_user_preference_context(
        self,
        preferred_tone: Optional[str] = None,
        preferred_maintenance: Optional[str] = None,
        preferred_language: Optional[str] = None,
        tone_diversity: int = 0
    ) -> Dict[str, Any]:
        """
        사용자 선호도 컨텍스트 추출 (Echo Feedback 지원)
        
        실제 통계 반영:
        - Normal 톤: 69.3%
        - Formal 톤: 19.6%
        - 평균 톤 다양성: 1.8가지
        """
        try:
            # Context Echo: 톤 다양성 분석
            is_single_tone_user = tone_diversity <= 1
            needs_exploration = tone_diversity <= 2  # 대부분 1-2가지만 사용
            
            return {
                'preferred_tone': preferred_tone,
                'preferred_maintenance': preferred_maintenance,
                'preferred_language': preferred_language,
                'tone_diversity': tone_diversity,
                'is_single_tone_user': is_single_tone_user,
                'needs_exploration': needs_exploration,
                'has_preferences': preferred_tone is not None,
            }
        except Exception as e:
            self.logger.error(f"Failed to extract user preference context: {e}", exc_info=True)
            return self._get_default_user_preference_context()

    def _get_default_user_preference_context(self) -> Dict[str, Any]:
        return {
            'preferred_tone': None,
            'preferred_maintenance': None,
            'preferred_language': None,
            'tone_diversity': 0,
            'is_single_tone_user': False,
            'needs_exploration': True,
            'has_preferences': False,
        }

    # =========================================================
    # Full Context Extraction (전체 컨텍스트 추출)
    # =========================================================

    def extract_full_context(self, **kwargs) -> Dict[str, Any]:
        """
        모든 컨텍스트 피처를 한 번에 추출
        
        지원하는 추천 시스템 컨셉:
        - Ambient Recommendation: 환경 기반 배치
        - Echo Feedback: 사용자 기억
        - Temporal Flow: 시간대별 추천
        - Context Echo: 다양성 유도
        """
        try:
            return {
                'temporal': self.extract_temporal_context(
                    timestamp=kwargs.get('timestamp')
                ),
                'device': self.extract_device_context(
                    browser=kwargs.get('browser'),
                    os=kwargs.get('os'),
                    device_id=kwargs.get('device_id')
                ),
                'location': self.extract_location_context(
                    country_code=kwargs.get('country_code'),
                    city=kwargs.get('city'),
                    timezone=kwargs.get('timezone')
                ),
                'session': self.extract_session_context(
                    events_in_session=kwargs.get('events_in_session', 0),
                    session_duration_minutes=kwargs.get('session_duration_minutes', 0.0),
                    last_event_name=kwargs.get('last_event_name')
                ),
                'user_preference': self.extract_user_preference_context(
                    preferred_tone=kwargs.get('preferred_tone'),
                    preferred_maintenance=kwargs.get('preferred_maintenance'),
                    preferred_language=kwargs.get('preferred_language'),
                    tone_diversity=kwargs.get('tone_diversity', 0)
                ),
            }
        except Exception as e:
            self.logger.error(f"Failed to extract full context: {e}", exc_info=True)
            return {
                'temporal': self._get_default_temporal_context(),
                'device': self._get_default_device_context(),
                'location': self._get_default_location_context(),
                'session': self._get_default_session_context(),
                'user_preference': self._get_default_user_preference_context(),
            }


# =========================================================
# Demo (Optional)
# =========================================================

if __name__ == "__main__":
    from datetime import datetime
    import json
    logging.basicConfig(level=logging.INFO)

    extractor = ContextFeatureExtractor()
    print("="*60)
    print("🎯 Context Feature Extraction Demo")
    print("="*60)

    # 1. Temporal Context (새벽 사용자)
    print("\n🕐 Temporal Context (새벽 5시):")
    dawn_time = datetime.now().replace(hour=5, minute=30)
    temporal = extractor.extract_temporal_context(timestamp=dawn_time)
    print(json.dumps(temporal, indent=2, ensure_ascii=False))

    # 2. Device Context (Chrome on Mac)
    print("\n💻 Device Context (Chrome on Mac):")
    device = extractor.extract_device_context(
        browser="Chrome", 
        os="Mac OS X", 
        device_id="mac-001"
    )
    print(json.dumps(device, indent=2, ensure_ascii=False))

    # 3. Location Context (한국 사용자)
    print("\n🌍 Location Context (Seoul, Korea):")
    location = extractor.extract_location_context(
        country_code="KR", 
        city="Seoul", 
        timezone="Asia/Seoul"
    )
    print(json.dumps(location, indent=2, ensure_ascii=False))

    # 4. Session Context (파워유저)
    print("\n📊 Session Context (Power User):")
    session = extractor.extract_session_context(
        events_in_session=75, 
        session_duration_minutes=35.2
    )
    print(json.dumps(session, indent=2, ensure_ascii=False))

    # 5. User Preference Context
    print("\n🎨 User Preference Context:")
    preference = extractor.extract_user_preference_context(
        preferred_tone="normal",
        preferred_maintenance="moderate",
        preferred_language="Korean",
        tone_diversity=2
    )
    print(json.dumps(preference, indent=2, ensure_ascii=False))

    # 6. Full Context (통합)
    print("\n🎯 Full Context (오전 7시 파워유저):")
    full = extractor.extract_full_context(
        timestamp=datetime.now().replace(hour=7, minute=30),
        browser="Chrome",
        os="Mac OS X",
        country_code="KR",
        city="Seoul",
        events_in_session=55,
        session_duration_minutes=25.5,
        preferred_tone="normal",
        tone_diversity=2
    )
    print(json.dumps(full, indent=2, ensure_ascii=False))
    
    print("\n" + "="*60)
    print("✅ Demo completed!")
    print("="*60)