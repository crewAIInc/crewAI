import ccxt
import pandas as pd
import ta
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from ..schemas.technical_analyst import (
    TechnicalAnalystInput, SystemHeader, MarketSnapshot, Candle,
    TechnicalIndicators, M15BarAnalysis, EmaCluster, MomentumIndicators,
    VolatilityIndicators, HtfContext, LearningBridge, BarStatus,
    EmaAlignment, SlopeState, RsiState, MacdState, DivergenceType,
    TrendStrength, BbState, VolatilityRank, RunMode
)

class TechnicalAnalystTools:
    """
    技术分析工具类：负责从交易所抓取数据并计算 Pydantic 实体所需的所有技术参数
    """

    def __init__(self, exchange_id: str = 'binance'):
        self.exchange = getattr(ccxt, exchange_id)()

    def fetch_ohlcv_df(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        """从交易所获取 K 线数据并转换为 DataFrame"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Asia/Tokyo')
            return df
        except Exception as e:
            print(f"Error fetching OHLCV for {symbol} {timeframe}: {e}")
            return pd.DataFrame()

    def get_bar_status(self, open_p, high, low, close) -> BarStatus:
        """识别单根 K 线形态"""
        body_size = abs(close - open_p)
        total_size = high - low if high > low else 0.00000001
        upper_shadow = high - max(open_p, close)
        lower_shadow = min(open_p, close) - low
        
        is_bullish = close > open_p
        
        # 简单逻辑识别形态
        if body_size / total_size > 0.6:
            return BarStatus.STRONG_BULLISH if is_bullish else BarStatus.STRONG_BEARISH
        if lower_shadow / total_size > 0.6 and body_size / total_size < 0.3:
            return BarStatus.BULLISH_PINBAR
        if upper_shadow / total_size > 0.6 and body_size / total_size < 0.3:
            return BarStatus.BEARISH_PINBAR
        
        return BarStatus.BULLISH if is_bullish else BarStatus.BEARISH

    def calculate_indicators_m15(self, df: pd.DataFrame) -> M15BarAnalysis:
        """计算 15 分钟级别的核心指标"""
        # ── 1. 集中计算所有指标列，避免快照丢列 ──
        # EMA
        df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        df['ema_200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()

        # RSI
        df['RSI_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

        # MACD
        macd_ind = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd_ind.macd()
        df['MACD_signal'] = macd_ind.macd_signal()

        # ADX
        df['ADX_14'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()

        # Bollinger Bands
        bb_ind = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['BB_upper'] = bb_ind.bollinger_hband()
        df['BB_middle'] = bb_ind.bollinger_mavg()
        df['BB_lower'] = bb_ind.bollinger_lband()

        # ATR
        df['ATR_14'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

        # ── 2. 所有列就绪后再取快照 ──
        last = df.iloc[-1]
        prev = df.iloc[-2]

        # ── 3. EMA 排列 ──
        alignment = EmaAlignment.TANGLED
        if last['ema_20'] > last['ema_50'] > last['ema_200']:
            alignment = EmaAlignment.PERFECT_BULLISH if last['close'] > last['ema_20'] else EmaAlignment.BULLISH_WEAK
        elif last['ema_20'] < last['ema_50'] < last['ema_200']:
            alignment = EmaAlignment.PERFECT_BEARISH if last['close'] < last['ema_20'] else EmaAlignment.BEARISH_WEAK

        # 斜率 (简单根据最近两根线判断)
        slope_val = (last['ema_20'] - prev['ema_20']) / last['ema_20'] * 100
        slope = SlopeState.FLAT
        if slope_val > 0.02: slope = SlopeState.STEEP_UP
        elif slope_val < -0.02: slope = SlopeState.STEEP_DOWN

        ema_cluster = EmaCluster(
            alignment=alignment,
            ema_20=round(last['ema_20'], 2),
            ema_50=round(last['ema_50'], 2),
            ema_200=round(last['ema_200'], 2),
            price_dist_ema20=f"{round((last['close'] - last['ema_20']) / last['ema_20'] * 100, 2)}%",
            ema_slope=slope
        )

        # ── 4. 动能指标 ──
        rsi_val = last['RSI_14']
        rsi_state = RsiState.NEUTRAL_LOW
        if rsi_val > 70: rsi_state = RsiState.OVERBOUGHT
        elif rsi_val > 50: rsi_state = RsiState.NEUTRAL_HIGH
        elif rsi_val < 30: rsi_state = RsiState.OVERSOLD

        macd_state = MacdState.ABOVE_ZERO if last['MACD'] > 0 else MacdState.BELOW_ZERO
        # 金叉死叉逻辑
        if last['MACD'] > last['MACD_signal'] and prev['MACD'] <= prev['MACD_signal']:
            macd_state = MacdState.BULLISH_GOLDEN_CROSS
        elif last['MACD'] < last['MACD_signal'] and prev['MACD'] >= prev['MACD_signal']:
            macd_state = MacdState.BEARISH_DEATH_CROSS

        adx_val = last['ADX_14']
        trend_str = TrendStrength.WEAK
        if adx_val > 25: trend_str = TrendStrength.TRENDING
        if adx_val > 40: trend_str = TrendStrength.STRONG_TREND

        momentum = MomentumIndicators(
            rsi_14=round(rsi_val, 2),
            rsi_state=rsi_state,
            macd_state=macd_state,
            divergence=DivergenceType.NONE,
            adx=round(adx_val, 2),
            trend_strength=trend_str
        )

        # ── 5. 波动率指标 ──
        bb_width = (last['BB_upper'] - last['BB_lower']) / last['BB_middle']
        prev_bb_width = (prev['BB_upper'] - prev['BB_lower']) / prev['BB_middle']
        
        bb_state = BbState.STABLE
        if bb_width > prev_bb_width * 1.1: bb_state = BbState.EXPANDING
        elif bb_width < prev_bb_width * 0.9: bb_state = BbState.SQUEEZING

        # 波动率排名 (简单对比最近 100 根线的 ATR 均值)
        avg_atr = df['ATR_14'].tail(100).mean()
        vol_rank = VolatilityRank.NORMAL
        if last['ATR_14'] > avg_atr * 1.3: vol_rank = VolatilityRank.HIGH
        elif last['ATR_14'] < avg_atr * 0.7: vol_rank = VolatilityRank.LOW

        volatility = VolatilityIndicators(
            bb_state=bb_state,
            bb_width=round(bb_width, 4),
            atr=round(last['ATR_14'], 2),
            volatility_rank=vol_rank
        )

        return M15BarAnalysis(
            ema_cluster=ema_cluster,
            momentum=momentum,
            volatility=volatility
        )

    def get_htf_context(self, df_d1: pd.DataFrame, df_w1: pd.DataFrame) -> HtfContext:
        """获取大周期上下文"""
        last_d1 = df_d1.iloc[-1]
        last_w1 = df_w1.iloc[-1]
        
        # 计算日线 RSI
        df_d1['RSI_14'] = ta.momentum.RSIIndicator(df_d1['close'], window=14).rsi()
        d1_rsi = df_d1.iloc[-1]['RSI_14']
        
        # 日线趋势
        ema_200 = ta.trend.EMAIndicator(df_d1['close'], window=200).ema_indicator().iloc[-1]
        d1_trend = BarStatus.BULLISH if last_d1['close'] > ema_200 else BarStatus.BEARISH
        
        # 周线支撑 (简单取最近 20 周最低)
        w1_key_support = df_w1['low'].tail(20).min()
        
        # 市场结构 (简单判断 Higher High)
        recent_highs = df_d1['high'].tail(50).rolling(window=10).max()
        market_structure = "Higher_High" if recent_highs.iloc[-1] > recent_highs.iloc[-10] else "Range"

        return HtfContext(
            d1_trend=d1_trend,
            d1_rsi=round(d1_rsi, 2),
            w1_trend="Macro_Uptrend" if last_w1['close'] > df_w1['close'].shift(20).iloc[-1] else "Macro_Downtrend",
            w1_key_support=round(w1_key_support, 2),
            market_structure=market_structure
        )

    def generate_input(self, symbol: str, request_id: str, learning_bridge: LearningBridge) -> TechnicalAnalystInput:
        """
        全自动生成技术分析师的输入数据
        """
        # 1. 抓取各周期数据
        df_m5 = self.fetch_ohlcv_df(symbol, '5m', limit=100)
        df_m15 = self.fetch_ohlcv_df(symbol, '15m', limit=1000)
        df_d1 = self.fetch_ohlcv_df(symbol, '1d', limit=300)
        df_w1 = self.fetch_ohlcv_df(symbol, '1w', limit=100)

        # 2. 构建 MarketSnapshot
        m5_candles = []
        for _, row in df_m5.tail(5).iterrows():
            m5_candles.append(Candle(
                t=row['datetime'].strftime('%H:%M'),
                o=row['open'], h=row['high'], l=row['low'], c=row['close'], v=row['volume'],
                bar_status=self.get_bar_status(row['open'], row['high'], row['low'], row['close'])
            ))

        m15_candles = []
        for _, row in df_m15.tail(5).iterrows():
            # 简单计算 vol_ratio (对比过去 20 根线均值)
            avg_vol = df_m15['volume'].tail(20).mean()
            m15_candles.append(Candle(
                t=row['datetime'].strftime('%H:%M'),
                o=row['open'], h=row['high'], l=row['low'], c=row['close'], v=row['volume'],
                vol_ratio=round(row['volume'] / avg_vol, 2)
            ))

        snapshot = MarketSnapshot(m5_candles=m5_candles, m15_candles=m15_candles)

        # 3. 计算多维指标层
        m15_analysis = self.calculate_indicators_m15(df_m15)
        htf_context = self.get_htf_context(df_d1, df_w1)
        indicators = TechnicalIndicators(
            m15_1000_bars=m15_analysis,
            htf_context=htf_context
        )

        # 4. 组装完整输入
        header = SystemHeader(
            request_id=request_id,
            symbol=symbol,
            timestamp=datetime.now(),
            run_mode=RunMode.LIVE
        )

        return TechnicalAnalystInput(
            system_header=header,
            market_snapshot=snapshot,
            technical_indicators=indicators,
            learning_bridge=learning_bridge
        )
