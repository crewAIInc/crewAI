from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional, List
from sqlalchemy import BIGINT, VARCHAR, INT, TEXT, DECIMAL, TIMESTAMP, JSON, Boolean, ForeignKey, Enum, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# --- 基础类定义 ---
class Base(DeclarativeBase):
    """所有数据库模型的基类"""
    pass

# --- 1. 枚举类型定义 (与数据库 ENUM 对应) ---

class FinalActionEnum(str, PyEnum):
    EXECUTE_LONG = "EXECUTE_LONG"   # 执行开多
    EXECUTE_SHORT = "EXECUTE_SHORT" # 执行开空
    STAY_OUT = "STAY_OUT"           # 观望/跳过

class AgentNameEnum(str, PyEnum):
    AI_1 = "AI_1" # 技术分析师
    AI_2 = "AI_2" # 情绪分析师
    AI_3 = "AI_3" # 首席决策师

class AiStopStatusEnum(str, PyEnum):
    AI1_STOP = "ai1_stop" # 在AI 1处停止
    AI3_STOP = "ai3_stop" # 在AI 3处停止

class TradeStatusEnum(str, PyEnum):
    OPEN = "OPEN"           # 持仓中
    CLOSED = "CLOSED"       # 已平仓
    CANCELLED = "CANCELLED" # 已取消

class CloseReasonEnum(str, PyEnum):
    TAKE_PROFIT = "TAKE_PROFIT" # 止盈
    STOP_LOSS = "STOP_LOSS"     # 止损
    MANUAL_CLOSE = "MANUAL_CLOSE" # 手动平仓

# --- 2. 数据库实体类定义 ---

class AgentDecision(Base):
    """
    核心表：agent_decisions（决策流水表）
    记录 AI 3 产生的每一个判决，是系统的“大脑日志”。
    """
    __tablename__ = "agent_decisions"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True, comment="主键，自增")
    request_id: Mapped[str] = mapped_column(VARCHAR(64), unique=True, index=True, nullable=False, comment="请求唯一标识")
    symbol: Mapped[str] = mapped_column(VARCHAR(16), index=True, nullable=False, comment="交易对(如 BTCUSDT)")
    final_action: Mapped[FinalActionEnum] = mapped_column(Enum(FinalActionEnum), nullable=False, comment="最终执行动作")
    total_score: Mapped[int] = mapped_column(INT, nullable=False, comment="最终100分制总分")
    
    # 关联字段：虽然是BIGINT，但在SQLAlchemy中定义为外键关联
    ai1_id: Mapped[Optional[int]] = mapped_column(ForeignKey("agent_conversations.id"), comment="关联AI 1的原始对话记录")
    ai2_id: Mapped[Optional[int]] = mapped_column(ForeignKey("agent_conversations.id"), comment="关联AI 2的原始对话记录")
    ai3_id: Mapped[Optional[int]] = mapped_column(ForeignKey("agent_conversations.id"), comment="关联AI 3的原始对话记录")
    
    core_content: Mapped[Optional[str]] = mapped_column(TEXT, comment="核心摘要/沉鱼字段")
    attendance_impact: Mapped[Optional[str]] = mapped_column(VARCHAR(32), comment="记录是否为考勤强制开单")
    ai_stop_status: Mapped[Optional[AiStopStatusEnum]] = mapped_column(Enum(AiStopStatusEnum), nullable=True, comment="在哪个AI阶段停止了")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, server_default=func.now(), comment="生成时间")

    # 关系映射
    conversations: Mapped[List["AgentConversation"]] = relationship("AgentConversation", back_populates="decision", foreign_keys="[AgentConversation.decision_id]")
    trade: Mapped[Optional["Trade"]] = relationship("Trade", back_populates="decision")

class AgentConversation(Base):
    """
    记录表：agent_conversations (AI 原始对话记录表)
    存储喂给 AI 的完整输入和其返回的完整输出。
    """
    __tablename__ = "agent_conversations"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True, comment="主键")
    decision_id: Mapped[int] = mapped_column(BIGINT, ForeignKey("agent_decisions.id"), index=True, comment="关联决策ID")
    agent_name: Mapped[AgentNameEnum] = mapped_column(Enum(AgentNameEnum), nullable=False, comment="智能体名称")
    
    # JSON 字段：MySQL 5.7+ 支持的原生 JSON 类型
    input_json: Mapped[dict] = mapped_column(JSON, nullable=False, comment="喂给该 AI 的完整输入 JSON")
    output_json: Mapped[dict] = mapped_column(JSON, nullable=False, comment="该 AI 返回的完整输出 JSON")
    
    core_content: Mapped[Optional[str]] = mapped_column(TEXT, comment="存储重要信息(如 Telegram 推送文本)")
    token_usage: Mapped[Optional[int]] = mapped_column(INT, comment="本次消耗的 Token 数量")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, server_default=func.now(), comment="记录时间")

    # 关系映射
    decision: Mapped["AgentDecision"] = relationship("AgentDecision", back_populates="conversations", foreign_keys=[decision_id])

class Trade(Base):
    """
    核心表：trades（订单实盘表）
    记录真实的交易执行情况及盈亏回填。
    """
    __tablename__ = "trades"

    trade_id: Mapped[str] = mapped_column(VARCHAR(64), primary_key=True, comment="交易所订单ID")
    parent_decision_id: Mapped[int] = mapped_column(BIGINT, ForeignKey("agent_decisions.id"), index=True, comment="关联决策ID")
    status: Mapped[TradeStatusEnum] = mapped_column(Enum(TradeStatusEnum), default=TradeStatusEnum.OPEN, comment="订单状态")
    
    # 价格与仓位：使用 DECIMAL 确保精度
    entry_price: Mapped[float] = mapped_column(DECIMAL(18, 8), nullable=False, comment="实际入场价格")
    exit_price: Mapped[Optional[float]] = mapped_column(DECIMAL(18, 8), comment="实际离场价格")
    position_size: Mapped[float] = mapped_column(DECIMAL(18, 8), nullable=False, comment="实际成交数量")
    leverage: Mapped[int] = mapped_column(INT, nullable=False, comment="使用的杠杆倍数")
    
    realized_pnl: Mapped[Optional[float]] = mapped_column(DECIMAL(18, 8), comment="实际盈亏金额")
    rr_ratio_actual: Mapped[Optional[float]] = mapped_column(DECIMAL(5, 2), comment="实际盈亏比")
    close_reason: Mapped[Optional[CloseReasonEnum]] = mapped_column(Enum(CloseReasonEnum), comment="平仓原因")
    
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, server_default=func.now(), comment="开单时间")
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP, server_default=func.now(), onupdate=func.now(), comment="更新时间")

    # 关系映射
    decision: Mapped["AgentDecision"] = relationship("AgentDecision", back_populates="trade")

class SystemEvolution(Base): # 注意：这里建议使用 Base，但字段名按你要求
    """
    辅助表：system_evolution（进化/纠错表）
    用于存储对 AI 的改进指令，实现系统的自我进化。
    """
    __tablename__ = "system_evolution"

    instruction_id: Mapped[str] = mapped_column(VARCHAR(32), primary_key=True, comment="指令ID(如 EVO-001)")
    target_agent: Mapped[AgentNameEnum] = mapped_column(Enum(AgentNameEnum), nullable=False, comment="目标智能体")
    error_case_id: Mapped[Optional[str]] = mapped_column(VARCHAR(64), ForeignKey("trades.trade_id"), comment="关联导致错误的订单ID")
    instruction_content: Mapped[str] = mapped_column(TEXT, nullable=False, comment="具体纠错指令内容")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, comment="是否仍在生效")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, server_default=func.now(), comment="创建时间")

class AgentPerformanceMetric(Base):
    """
    看板表：agent_performance_metrics (AI 表现统计表)
    每天定时计算，用于动态调整 AI 权重。
    """
    __tablename__ = "agent_performance_metrics"

    agent_id: Mapped[str] = mapped_column(VARCHAR(32), primary_key=True, comment="智能体标识(AI_1, AI_2, AI_3)")
    win_rate_7d: Mapped[float] = mapped_column(DECIMAL(5, 2), default=0.00, comment="过去7天胜率")
    avg_score_bias: Mapped[float] = mapped_column(DECIMAL(5, 2), default=0.00, comment="评分偏差(是否习惯性给高分)")
    consecutive_skips: Mapped[int] = mapped_column(INT, default=0, comment="累计跳单次数")
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP, server_default=func.now(), onupdate=func.now(), comment="统计更新时间")
