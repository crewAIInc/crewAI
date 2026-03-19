from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime

# --- 1. 枚举定义 (Enums) ---

class RunMode(str, Enum):
    """运行模式"""
    LIVE = "Live"        # 实盘
    SANDBOX = "Sandbox"  # 沙盒/模拟
    BACKTEST = "Backtest" # 回测

class BarStatus(str, Enum):
    """K线形态/状态"""
    STRONG_BULLISH = "Strong_Bullish"   # 强多头
    BULLISH = "Bullish"                 # 多头
    NEUTRAL = "Neutral"                 # 中性
    BEARISH = "Bearish"                 # 空头
    STRONG_BEARISH = "Strong_Bearish"   # 强空头
    BEARISH_PINBAR = "Bearish_Pinbar"   # 空头看跌针
    BULLISH_PINBAR = "Bullish_PINBAR"   # 多头看涨针

class EmaAlignment(str, Enum):
    """EMA 均线排列状态"""
    PERFECT_BULLISH = "Perfect_Bullish" # 完美多头排列
    BULLISH_WEAK = "Bullish_Weak"       # 弱多头
    PERFECT_BEARISH = "Perfect_Bearish" # 完美空头排列
    BEARISH_WEAK = "Bearish_Weak"       # 弱空头
    TANGLED = "Tangled"                 # 缠绕/纠结

class SlopeState(str, Enum):
    """均线斜率"""
    STEEP_UP = "Steep_Up"     # 陡峭向上
    FLAT = "Flat"             # 平缓
    STEEP_DOWN = "Steep_Down" # 陡峭向下

class RsiState(str, Enum):
    """RSI 状态"""
    OVERSOLD = "Oversold"         # 超卖
    NEUTRAL_LOW = "Neutral_Low"   # 中性偏低
    NEUTRAL_HIGH = "Neutral_High" # 中性偏高
    OVERBOUGHT = "Overbought"     # 超买

class MacdState(str, Enum):
    """MACD 状态"""
    GOLDEN_CROSS = "Golden_Cross"                 # 金叉
    BULLISH_GOLDEN_CROSS = "Bullish_Golden_Cross" # 多头金叉
    DEATH_CROSS = "Death_Cross"                   # 死叉
    BEARISH_DEATH_CROSS = "Bearish_Death_Cross"   # 空头死叉
    ABOVE_ZERO = "Above_Zero"                     # 零轴上方
    BELOW_ZERO = "Below_Zero"                     # 零轴下方

class DivergenceType(str, Enum):
    """背离类型"""
    NONE = "None"               # 无背离
    BULLISH_REGULAR = "Bullish_Regular" # 常规底背离
    BEARISH_REGULAR = "Bearish_Regular" # 常规顶背离
    BULLISH_HIDDEN = "Bullish_Hidden"   # 隐藏底背离
    BEARISH_HIDDEN = "Bearish_Hidden"   # 隐藏顶背离

class TrendStrength(str, Enum):
    """趋势强度"""
    WEAK = "Weak"               # 弱
    TRENDING = "Trending"       # 趋势中
    STRONG_TREND = "Strong_Trend" # 强趋势

class BbState(str, Enum):
    """布林带状态"""
    SQUEEZING = "Squeezing" # 挤压/收口
    EXPANDING = "Expanding" # 扩张/开口
    STABLE = "Stable"       # 稳定

class VolatilityRank(str, Enum):
    """波动率排名"""
    LOW = "Low"
    NORMAL = "Normal"
    HIGH = "High"

class LiquidationRiskLevel(str, Enum):
    """清算风险等级（基于资金费率+持仓变化的代理指标）"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class TradingAction(str, Enum):
    """交易动作建议"""
    STRONG_BUY = "Strong_Buy"
    BUY = "Buy"
    NEUTRAL = "Neutral"
    SELL = "Sell"
    STRONG_SELL = "Strong_Sell"

class DecisionOutcome(str, Enum):
    """决策结果状态"""
    APPROVED = "APPROVED"                         # 已批准
    REJECTED = "REJECTED"                         # 已拒绝
    ADJUSTED = "ADJUSTED"                         # 已调整
    REJECTED_BY_STRATEGIST = "REJECTED_BY_STRATEGIST" # 被策略师拒绝

class PerformanceStatus(str, Enum):
    """预测表现评价"""
    CORRECT = "CORRECT" # 正确
    WRONG = "WRONG"     # 错误
    NEUTRAL = "NEUTRAL" # 中性

class TradeResult(str, Enum):
    """交易结果"""
    PROFIT = "PROFIT" # 盈利
    LOSS = "LOSS"     # 亏损

class TimingState(str, Enum):
    """时机评价"""
    OPTIMAL = "OPTIMAL"           # 最佳
    LATE = "LATE"                 # 偏晚
    OVEREXTENDED = "OVEREXTENDED" # 过度延伸
    EARLY = "EARLY"               # 偏早

class HtfAlignment(str, Enum):
    """大周期对齐状态"""
    ALIGNED = "ALIGNED"   # 已对齐
    CONFLICT = "CONFLICT" # 冲突
    NEUTRAL = "NEUTRAL"   # 中性

# --- 2. 输入实体类 (Input Models) ---

class SystemHeader(BaseModel):
    """系统头信息"""
    request_id: str = Field(..., description="请求唯一ID")
    symbol: str = Field(..., description="交易对符号")
    timestamp: datetime = Field(..., description="请求时间戳")
    run_mode: RunMode = Field(..., description="运行模式")

class Candle(BaseModel):
    """K线基础实体"""
    t: str = Field(..., description="时间(格式如 20:25)")
    o: float = Field(..., description="开盘价")
    h: float = Field(..., description="最高价")
    l: float = Field(..., description="最低价")
    c: float = Field(..., description="收盘价")
    v: float = Field(..., description="成交量")
    bar_status: Optional[BarStatus] = Field(None, description="K线形态状态")
    vol_ratio: Optional[float] = Field(None, description="成交量比率")

class MarketSnapshot(BaseModel):
    """实时量价快照层"""
    current_price: float = Field(..., description="当前最新价格")
    price_change_24h: str = Field(..., description="24小时涨跌幅(如 +2.35%)")
    m5_candles: List[Candle] = Field(..., description="5分钟级别K线列表")
    m15_candles: List[Candle] = Field(..., description="15分钟级别K线列表")
    h1_candles: List[Candle] = Field(..., description="1小时级别K线列表")

class EmaCluster(BaseModel):
    """EMA 均线簇指标"""
    alignment: EmaAlignment = Field(..., description="均线排列状态")
    ema_20: float = Field(..., description="20日均线值")
    ema_50: float = Field(..., description="50日均线值")
    ema_200: float = Field(..., description="200日均线值")
    price_dist_ema20: str = Field(..., description="价格距离EMA20的百分比距离")
    ema_slope: SlopeState = Field(..., description="均线斜率状态")

class MomentumIndicators(BaseModel):
    """动量指标"""
    rsi_14: float = Field(..., description="RSI(14)值")
    rsi_state: RsiState = Field(..., description="RSI所处状态")
    macd_state: MacdState = Field(..., description="MACD状态")
    divergence: DivergenceType = Field(..., description="背离情况")
    adx: float = Field(..., description="ADX趋势强度值")
    trend_strength: TrendStrength = Field(..., description="趋势强度分类")

class VolatilityIndicators(BaseModel):
    """波动率指标"""
    bb_state: BbState = Field(..., description="布林带开口状态")
    bb_width: float = Field(..., description="布林带宽度")
    atr: float = Field(..., description="ATR波动值")
    volatility_rank: VolatilityRank = Field(..., description="基于历史的波动率排名")

class M15BarAnalysis(BaseModel):
    """15分钟级别多维度分析"""
    ema_cluster: EmaCluster
    momentum: MomentumIndicators
    volatility: VolatilityIndicators

class HtfContext(BaseModel):
    """大周期(HTF)上下文层 — 基于 4h + 1h"""
    h4_trend: BarStatus = Field(..., description="4小时线趋势")
    h4_rsi: float = Field(..., description="4小时RSI")
    h1_trend: BarStatus = Field(..., description="1小时线趋势")
    h1_rsi: float = Field(..., description="1小时RSI")
    h4_key_support: float = Field(..., description="4小时关键支撑位")
    h4_key_resistance: float = Field(..., description="4小时关键阻力位")
    market_structure: str = Field(..., description="市场结构(如 Higher_High)")

class DerivativesContext(BaseModel):
    """合约市场衍生品上下文"""
    funding_rate: float = Field(..., description="当前资金费率（小数，如 0.0001 = 0.01%）")
    funding_rate_pct: str = Field(..., description="资金费率百分比展示（如 +0.0100%）")
    next_funding_time: Optional[str] = Field(None, description="下一次资金费结算时间（交易所时区字符串）")
    open_interest_value: float = Field(..., description="当前未平仓名义价值")
    open_interest_change_24h: str = Field(..., description="未平仓量24h变化（如 +8.32%）")
    liquidation_risk_level: LiquidationRiskLevel = Field(..., description="清算风险等级")
    liquidation_risk_note: str = Field(..., description="清算风险简述（100字以内）")

class TechnicalIndicators(BaseModel):
    """多维指标层汇总"""
    m15_1000_bars: M15BarAnalysis
    htf_context: HtfContext
    derivatives_context: DerivativesContext

class LastRecommendation(BaseModel):
    """上次推荐记录(纠错参考)"""
    timestamp: datetime = Field(..., description="上次建议时间")
    your_signal: TradingAction = Field(..., description="当时给出的信号")
    decision_outcome: DecisionOutcome = Field(..., description="最终决策执行结果")
    reason: str = Field(..., description="被拒绝或调整的原因")
    market_performance_after: str = Field(..., description="建议后的市场表现描述")
    market_performance: PerformanceStatus = Field(..., description="预测是否正确")

class PnlFeedback(BaseModel):
    """近期盈亏反馈详情"""
    trade_id: str = Field(..., description="交易ID")
    signal: TradingAction = Field(..., description="当时信号")
    result: TradeResult = Field(..., description="交易结果")
    comment: str = Field(..., description="复盘评论")

class LearningBridge(BaseModel):
    """核心纠错层"""
    last_recommendation: LastRecommendation
    recent_pnl_feedback: List[PnlFeedback]
    evolution_instruction: str = Field(..., description="AI进化指令/改进建议")

class TechnicalAnalystInput(BaseModel):
    """技术分析师 AI 1：完整输入 JSON 实体"""
    model_config = ConfigDict(populate_by_name=True)

    system_header: SystemHeader
    market_snapshot: MarketSnapshot
    technical_indicators: TechnicalIndicators
    # learning_bridge: LearningBridge

# --- 3. 输出实体类 (Output Models) ---

class IndicatorSignal(BaseModel):
    """单个指标的信号评估"""
    name: str = Field(..., description="指标名称（如 EMA_Alignment, RSI_14, MACD, BB, ADX, H4_Trend, H1_Trend）")
    value: str = Field(..., description="当前值或状态描述")
    bias: str = Field(..., description="Bullish / Bearish / Neutral")
    weight: str = Field(..., description="High / Medium / Low — 该指标在当前市况下的权重")

class IndicatorConflict(BaseModel):
    """指标间矛盾"""
    indicator_a: str = Field(..., description="矛盾指标 A 名称")
    indicator_b: str = Field(..., description="矛盾指标 B 名称")
    detail: str = Field(..., description="矛盾具体表现")
    resolution: str = Field(..., description="权衡建议（50字以内）")

class HtfAlignmentResult(BaseModel):
    """大周期对齐分析结果"""
    status: HtfAlignment = Field(..., description="Aligned / Conflict / Neutral")
    h4_bias: str = Field(..., description="4H 周期偏向")
    h1_bias: str = Field(..., description="1H 周期偏向")
    m15_bias: str = Field(..., description="M15 周期偏向")
    detail: str = Field(..., description="对齐/冲突的具体说明（50字以内）")

class TechnicalIndicatorOutput(BaseModel):
    """Agent 2 技术指标分析师：结构化输出"""
    indicator_signals: List[IndicatorSignal] = Field(
        ..., description="各指标信号汇总列表（至少包含 EMA/RSI/MACD/BB/ADX/H4/H1 共7项）"
    )
    conflicts: List[IndicatorConflict] = Field(
        default_factory=list, description="指标间矛盾清单（无矛盾时为空列表）"
    )
    htf_alignment: HtfAlignmentResult = Field(
        ..., description="多周期对齐分析"
    )
    score: int = Field(
        ..., ge=-10, le=10,
        description="综合多空评分：-10(极度看空) ~ 0(中性) ~ +10(极度看多)"
    )
    overall_bias: str = Field(..., description="Bullish / Bearish / Neutral")
    confidence: int = Field(
        ..., ge=1, le=10, description="技术面分析置信度 1-10"
    )
    summary: str = Field(
        ..., description="一句话总结当前技术面状况（100字以内）"
    )
