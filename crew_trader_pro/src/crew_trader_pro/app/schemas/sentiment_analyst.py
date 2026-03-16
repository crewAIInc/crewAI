from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime

# --- 1. 枚举定义 (Enums) ---

class LsRatioStatus(str, Enum):
    """多空比状态"""
    BULL_EXTREME = "BULL_EXTREME"   # 极度看涨
    BULLISH_BIAS = "BULLISH_BIAS"   # 看涨偏向
    NEUTRAL = "NEUTRAL"             # 中性
    BEARISH_BIAS = "BEARISH_BIAS"   # 看跌偏向
    BEAR_EXTREME = "BEAR_EXTREME"   # 极度看跌

class FundingStatus(str, Enum):
    """资金费率状态"""
    STABLE = "STABLE"               # 稳定
    HIGH_LONG_COST = "HIGH_LONG_COST" # 多头成本高
    HIGH_SHORT_COST = "HIGH_SHORT_COST" # 空头成本高

class LiquidationSide(str, Enum):
    """清算方向"""
    LONG = "LONG"   # 多单清算
    SHORT = "SHORT" # 空单清算

class ImpactScore(str, Enum):
    """影响等级"""
    CRITICAL = "CRITICAL" # 至关重要/关键
    HIGH = "HIGH"         # 高
    MEDIUM = "MEDIUM"     # 中
    LOW = "LOW"           # 低

class MarketMood(str, Enum):
    """市场情绪/氛围"""
    EXTREME_FEAR = "EXTREME_FEAR" # 极度恐惧
    FEAR = "FEAR"                 # 恐惧
    NEUTRAL = "NEUTRAL"           # 中性
    GREED = "GREED"               # 贪婪
    EXTREME_GREED = "EXTREME_GREED" # 极度贪婪

class CrowdBias(str, Enum):
    """人群拥挤度/偏向"""
    LONG_CROWDED = "LONG_CROWDED"   # 多头拥挤
    SHORT_CROWDED = "SHORT_CROWDED" # 空头拥挤
    BALANCED = "BALANCED"           # 平衡

class LiquidationRisk(str, Enum):
    """清算/插针风险"""
    DOWNWARD_PIN = "DOWNWARD_PIN"     # 向下插针风险
    UPWARD_SQUEEZE = "UPWARD_SQUEEZE" # 向上轧空风险
    STABLE = "STABLE"                 # 稳定

class ContrarianIndicator(str, Enum):
    """逆向指标建议"""
    PROCEED = "PROCEED" # 继续执行
    CAUTION = "CAUTION" # 警惕/注意
    REVERSE = "REVERSE" # 反向操作建议

class WhaleAction(str, Enum):
    """大户/巨鲸动向"""
    ACCUMULATING = "ACCUMULATING" # 增持/吸筹
    DISTRIBUTING = "DISTRIBUTING" # 减持/派发
    NEUTRAL = "NEUTRAL"           # 中性

class DataFreshness(str, Enum):
    """数据新鲜度"""
    REAL_TIME = "REAL_TIME"   # 实时
    DELAYED = "DELAYED"       # 延迟
    HISTORICAL = "HISTORICAL" # 历史数据

# --- 2. 输入实体类 (Input Models) ---

class SentimentSystemHeader(BaseModel):
    """系统头信息"""
    request_id: str = Field(..., description="请求唯一ID")
    symbol: str = Field(..., description="交易对符号")
    timestamp: datetime = Field(..., description="请求时间戳")

class MarketSentimentIndices(BaseModel):
    """市场情绪指数"""
    fear_and_greed_index: int = Field(..., ge=0, le=100, description="恐慌贪婪指数 (0-100)")
    long_short_ratio_24h: float = Field(..., description="24小时多空人数比")
    ls_ratio_status: LsRatioStatus = Field(..., description="多空比状态描述")

class LiquidationVol(BaseModel):
    """24小时爆仓金额"""
    longs: float = Field(..., description="多单爆仓总额")
    shorts: float = Field(..., description="空单爆仓总额")

class HighDensityZone(BaseModel):
    """高密度爆仓区间"""
    price_range: str = Field(..., description="价格区间(如 71000-71300)")
    side: LiquidationSide = Field(..., description="清算方向")
    estimated_liquidation_vol: str = Field(..., description="预估清算规模(如 1.2B)")
    impact_score: ImpactScore = Field(..., description="影响权重")

class LiquidationHeatmap(BaseModel):
    """清算热力图分析"""
    current_price: float = Field(..., description="当前参考价格")
    high_density_zones: List[HighDensityZone] = Field(..., description="高密度爆仓区列表")

class ExchangeFlows(BaseModel):
    """交易所资金流与持仓分析"""
    funding_rate: float = Field(..., description="实时资金费率")
    predicted_funding_rate: float = Field(..., description="下一周期预测费率")
    funding_status: FundingStatus = Field(..., description="费率状态")
    open_interest_change_24h: str = Field(..., description="24小时持仓量变化百分比")
    liquidation_vol_24h: LiquidationVol = Field(..., description="24小时爆仓量统计")
    liquidation_heatmap: LiquidationHeatmap = Field(..., description="爆仓热力分布")

class HotNewsItem(BaseModel):
    """热门新闻实体"""
    time: str = Field(..., description="新闻发布时间")
    impact_level: ImpactScore = Field(..., description="新闻影响等级")
    headline: str = Field(..., description="新闻标题")
    source: str = Field(..., description="新闻来源")

class NewsAndSocial(BaseModel):
    """新闻与社交媒体舆情"""
    hot_news: List[HotNewsItem] = Field(..., description="实时热点新闻列表")
    social_volume_score: int = Field(..., ge=0, le=100, description="社交媒体讨论热度(0-100)")
    social_sentiment_polarity: float = Field(..., ge=-1, le=1, description="舆情极性(-1到1)")

class SentimentLearningBridge(BaseModel):
    """情绪分析师纠错层"""
    last_feedback: str = Field(..., description="上一次的反馈建议")
    evolution_instruction: str = Field(..., description="AI进化/改进指令")

class SentimentAnalystInput(BaseModel):
    """情绪分析师 AI 2：完整输入 JSON 实体"""
    model_config = ConfigDict(populate_by_name=True)

    system_header: SentimentSystemHeader
    market_sentiment_indices: MarketSentimentIndices
    exchange_flows: ExchangeFlows
    news_and_social: NewsAndSocial
    learning_bridge: SentimentLearningBridge

# --- 3. 输出实体类 (Output Models) ---

class SentimentAnalysisMetadata(BaseModel):
    """分析元数据"""
    agent_id: str = Field(..., description="Agent ID")
    timestamp: datetime = Field(..., description="分析生成时间")
    data_freshness: DataFreshness = Field(..., description="数据新鲜度")

class SentimentScoreCard(BaseModel):
    """情绪评分表"""
    score: int = Field(..., ge=0, le=25, description="综合得分(0-25)")
    funding_weight: int = Field(..., ge=0, le=10, description="费率健康度得分权重")
    social_weight: int = Field(..., ge=0, le=8, description="社交舆情得分权重")
    liquidation_weight: int = Field(..., ge=0, le=7, description="爆仓分布得分权重")

class SentimentQualitative(BaseModel):
    """定性情绪分析"""
    market_mood: MarketMood = Field(..., description="整体市场氛围")
    crowd_bias: CrowdBias = Field(..., description="人群拥挤度倾向")
    liquidation_risk: LiquidationRisk = Field(..., description="清算插针风险类型")

class SentimentCoreLogic(BaseModel):
    """核心逻辑层"""
    sentiment_thesis: str = Field(..., description="情绪分析核心论点")
    contrarian_indicator: ContrarianIndicator = Field(..., description="逆向指标操作建议")
    whale_action: WhaleAction = Field(..., description="大户/巨鲸行为识别")

class SentimentSelfCorrectionReport(BaseModel):
    """自我修正报告"""
    instruction_id: str = Field(..., description="指令ID")
    compliance: str = Field(..., description="指令执行合规说明")
    is_corrected: bool = Field(..., description="是否已完成修正")

class SentimentAnalystOutput(BaseModel):
    """情绪分析师 AI 2：完整输出 JSON 实体"""
    analysis_metadata: SentimentAnalysisMetadata
    sentiment_score_card: SentimentScoreCard
    sentiment_qualitative: SentimentQualitative
    core_logic: SentimentCoreLogic
    self_correction_report: SentimentSelfCorrectionReport
