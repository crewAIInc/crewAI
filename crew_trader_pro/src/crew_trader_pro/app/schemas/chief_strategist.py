from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime

# --- 1. 枚举定义 (Enums) ---

class FinalAction(str, Enum):
    """最终执行指令"""
    EXECUTE_LONG = "EXECUTE_LONG"   # 执行开多
    EXECUTE_SHORT = "EXECUTE_SHORT" # 执行开空
    STAY_OUT = "STAY_OUT"           # 观望/不操作

class ConfidenceRating(str, Enum):
    """信心评级"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class SystemNudge(str, Enum):
    """系统提示/催促逻辑"""
    NORMAL = "NORMAL"               # 正常
    URGENT_ACTION = "URGENT_ACTION" # 迫切行动(如连续跳单后)
    CAUTION_REQUIRED = "CAUTION_REQUIRED" # 需谨慎

class ActivityRequirement(str, Enum):
    """活跃度要求"""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"

class PrimaryDriver(str, Enum):
    """主导策略驱动因素"""
    HTF_RESONANCE = "HTF_RESONANCE"       # 大周期共振
    SENTIMENT_REVERSAL = "SENTIMENT_REVERSAL" # 情绪反转
    TECHNICAL_BREAKOUT = "TECHNICAL_BREAKOUT" # 技术突破
    ATTENDANCE_CORRECTION = "ATTENDANCE_CORRECTION" # 考勤修正驱动

class ComplianceStatus(str, Enum):
    """合规/校验状态"""
    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"

# --- 2. 输入实体类 (Input Models) ---

class DecisionHeader(BaseModel):
    """决策头信息"""
    request_id: str = Field(..., description="请求唯一ID")
    symbol: str = Field(..., description="交易对符号")
    current_price: float = Field(..., description="当前市场价格")
    timestamp: datetime = Field(..., description="决策请求时间戳")

class Ai1Signal(BaseModel):
    """AI 1 技术报告摘要"""
    score: int = Field(..., ge=0, le=25, description="AI 1 技术评分")
    signal_grade: str = Field(..., description="信号等级(如 STRONG_BUY)")
    timing: str = Field(..., description="入场时机评价")
    logic_summary: str = Field(..., description="核心逻辑摘要")
    self_correction_status: str = Field(..., description="自我修正执行状态")

class Ai2Signal(BaseModel):
    """AI 2 情绪报告摘要"""
    score: int = Field(..., ge=0, le=25, description="AI 2 情绪评分")
    market_mood: str = Field(..., description="市场氛围(如 GREED)")
    liquidation_risk: str = Field(..., description="清算风险提示")
    logic_summary: str = Field(..., description="核心逻辑摘要")

class InputSignals(BaseModel):
    """输入的智能体信号汇总"""
    ai1_technical_report: Ai1Signal
    ai2_sentiment_report: Ai2Signal

class RiskRewardAnalysis(BaseModel):
    """盈亏比/风险分析"""
    rr_score: int = Field(..., ge=0, le=25, description="盈亏比评分")
    potential_tp: float = Field(..., description="潜在止盈位")
    potential_sl: float = Field(..., description="潜在止损位")
    max_drawdown_limit: str = Field(..., description="最大回撤限制百分比")

class D1Core(BaseModel):
    """日线核心指标"""
    trend: str = Field(..., description="趋势状态")
    ema_200_dist: str = Field(..., description="距离日线EMA200的距离")
    rsi_14: float = Field(..., description="日线RSI值")
    market_structure: str = Field(..., alias="structure", description="市场结构")
    adx: float = Field(..., description="趋势强度ADX")

class W1Core(BaseModel):
    """周线核心指标"""
    macro_trend: str = Field(..., description="宏观趋势描述")
    candle_type: str = Field(..., description="周线K线形态")
    stoch_rsi: str = Field(..., description="周线随机RSI状态")

class HtfRawData(BaseModel):
    """原始大周期数据"""
    d1_core: D1Core
    w1_core: W1Core

class PastDecision(BaseModel):
    """历史决策记录"""
    trade_id: str
    decision: str
    result: str
    reasoning_flaw: str
    lesson: str

class MemoryContext(BaseModel):
    """历史记忆上下文"""
    past_decisions_review: List[PastDecision]
    avg_win_rate_l30: str = Field(..., description="最近30单平均胜率")

class HtfAlignmentAnalysis(BaseModel):
    """大周期对齐分析层"""
    htf_score: int = Field(..., ge=0, le=25, description="大周期共振评分")
    weekly_bias: str = Field(..., description="周线偏向")
    daily_structure: str = Field(..., description="日线结构")
    is_counter_trend: bool = Field(..., description="是否属于逆势操作")
    htf_raw_data: HtfRawData
    memory_context: MemoryContext

class QuantitativeMetrics(BaseModel):
    """量化指标层汇总"""
    risk_reward_analysis: RiskRewardAnalysis
    htf_alignment_analysis: HtfAlignmentAnalysis

class AttendanceSystem(BaseModel):
    """考勤与活跃度干预系统"""
    consecutive_skips: int = Field(..., description="连续未开单/跳单次数")
    attendance_score: int = Field(..., ge=0, le=100, description="考勤分数")
    system_nudge: SystemNudge = Field(..., description="系统干预指令")
    activity_requirement: ActivityRequirement = Field(..., description="当前活跃度要求")

class ChiefStrategistInput(BaseModel):
    """首席决策师 AI 3：完整输入 JSON 实体"""
    model_config = ConfigDict(populate_by_name=True)

    decision_header: DecisionHeader
    input_signals: InputSignals
    quantitative_metrics: QuantitativeMetrics
    attendance_system: AttendanceSystem

# --- 3. 输出实体类 (Output Models) ---

class FinalDecision(BaseModel):
    """最终决策结果"""
    action: FinalAction = Field(..., description="下单动作指令")
    confidence_rating: ConfidenceRating = Field(..., description="信心评级")
    total_weighted_score: int = Field(..., ge=0, le=100, description="综合总评分(满分100)")

class ExecutionParams(BaseModel):
    """执行参数层"""
    entry_price: float = Field(..., description="限价入场价格")
    take_profit: float = Field(..., description="止盈价格")
    stop_loss: float = Field(..., description="硬止损价格")
    position_size: str = Field(..., description="建议仓位百分比(如 15%)")
    leverage: str = Field(..., description="建议杠杆倍数(如 5x)")

class TelegramReport(BaseModel):
    """Telegram 推送报告模块"""
    report_title: str = Field(..., description="报告大标题")
    final_action: str = Field(..., description="结论文字(如 开多 (LONG))")
    leverage: str = Field(..., description="展示用的杠杆值")
    position_size: str = Field(..., description="展示用的仓位值")
    summary_text: str = Field(..., description="多维逻辑汇总话术")
    risk_reminder: str = Field(..., description="核心风险预警话术")
    data_reference: str = Field(..., description="原始数据锚点快照")
    status_icon: str = Field(..., description="视觉状态图标(🟩/🟥)")

class StrategicRationale(BaseModel):
    """战略理由分类"""
    primary_driver: PrimaryDriver = Field(..., description="本单主导驱动逻辑")
    attendance_impact: str = Field(..., description="考勤系统对决策的修正说明")

class ComplianceAudit(BaseModel):
    """合规性审计"""
    is_historical_consistent: bool = Field(..., description="是否与历史成功案例对标一致")
    evolution_validation: ComplianceStatus = Field(..., description="进化指令执行确认状态")

class ChiefStrategistOutput(BaseModel):
    """首席决策师 AI 3：完整输出 JSON 实体"""
    final_decision: FinalDecision
    execution_params: ExecutionParams
    telegram_report: TelegramReport
    strategic_rationale: StrategicRationale
    compliance_audit: ComplianceAudit
