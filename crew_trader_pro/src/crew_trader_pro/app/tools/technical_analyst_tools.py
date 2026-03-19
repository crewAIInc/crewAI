# import logging
import ccxt
import pandas as pd
import ta
from datetime import datetime
from typing import List, Dict, Any, Optional
from ..schemas.technical_analyst import (
    TechnicalAnalystInput, SystemHeader, MarketSnapshot, Candle,
    TechnicalIndicators, M15BarAnalysis, EmaCluster, MomentumIndicators,
    VolatilityIndicators, HtfContext, DerivativesContext, LearningBridge, BarStatus,
    EmaAlignment, SlopeState, RsiState, MacdState, DivergenceType,
    TrendStrength, BbState, VolatilityRank, LiquidationRiskLevel, RunMode
)

class TechnicalAnalystTools:
    """
    技术分析工具类：负责从交易所抓取数据并计算 Pydantic 实体所需的所有技术参数
    """

    def __init__(self, exchange_id: str = 'binance'):
        self.exchange = getattr(ccxt, exchange_id)({
            'options': {
                'defaultType': 'future',  # 永续合约
            }
        })

    @staticmethod
    def _format_pct(value: Optional[float], digits: int = 4) -> str:
        if value is None:
            return "N/A"
        return f"{'+' if value >= 0 else ''}{value:.{digits}f}%"

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

    def get_htf_context(self, df_h4: pd.DataFrame, df_h1: pd.DataFrame) -> HtfContext:
        """获取大周期上下文（基于 4h + 1h，适合日内交易）"""
        # 4h 趋势: EMA50 上方=多头
        ema_50_h4 = ta.trend.EMAIndicator(df_h4['close'], window=50).ema_indicator()
        last_h4 = df_h4.iloc[-1]
        h4_trend = BarStatus.BULLISH if last_h4['close'] > ema_50_h4.iloc[-1] else BarStatus.BEARISH

        # 4h RSI
        df_h4['RSI_14'] = ta.momentum.RSIIndicator(df_h4['close'], window=14).rsi()
        h4_rsi = df_h4.iloc[-1]['RSI_14']

        # 1h 趋势: EMA20 上方=多头
        ema_20_h1 = ta.trend.EMAIndicator(df_h1['close'], window=20).ema_indicator()
        last_h1 = df_h1.iloc[-1]
        h1_trend = BarStatus.BULLISH if last_h1['close'] > ema_20_h1.iloc[-1] else BarStatus.BEARISH

        # 1h RSI
        df_h1['RSI_14'] = ta.momentum.RSIIndicator(df_h1['close'], window=14).rsi()
        h1_rsi = df_h1.iloc[-1]['RSI_14']

        # 4h 支撑/阻力 (最近 50 根的最低/最高)
        h4_key_support = df_h4['low'].tail(50).min()
        h4_key_resistance = df_h4['high'].tail(50).max()

        # 市场结构 (简单判断 Higher High)
        recent_highs = df_h4['high'].tail(50).rolling(window=10).max()
        market_structure = "Higher_High" if recent_highs.iloc[-1] > recent_highs.iloc[-10] else "Range"

        return HtfContext(
            h4_trend=h4_trend,
            h4_rsi=round(h4_rsi, 2),
            h1_trend=h1_trend,
            h1_rsi=round(h1_rsi, 2),
            h4_key_support=round(h4_key_support, 2),
            h4_key_resistance=round(h4_key_resistance, 2),
            market_structure=market_structure
        )

    def get_derivatives_context(self, symbol: str, price_change_24h_pct: Optional[float]) -> DerivativesContext:
        """获取合约衍生品上下文（资金费率、未平仓量、清算风险代理）"""
        funding_rate = 0.0
        funding_rate_pct = "N/A"
        next_funding_time = None

        open_interest_value = 0.0
        oi_change_24h = "N/A"
        oi_change_24h_value: Optional[float] = None

        try:
            funding_data = self.exchange.fetch_funding_rate(symbol)
            funding_rate_raw = funding_data.get('fundingRate')
            if funding_rate_raw is not None:
                funding_rate = float(funding_rate_raw)
                funding_rate_pct = self._format_pct(funding_rate * 100, digits=4)
            next_funding_dt = funding_data.get('fundingDatetime') or funding_data.get('nextFundingTime')
            if next_funding_dt is not None:
                next_funding_time = str(next_funding_dt)
        except Exception:
            pass

        try:
            oi_data = self.exchange.fetch_open_interest(symbol)
            oi_val = oi_data.get('openInterestValue') or oi_data.get('openInterestAmount')
            if oi_val is not None:
                open_interest_value = float(oi_val)
        except Exception:
            pass

        try:
            oi_hist = self.exchange.fetch_open_interest_history(symbol, timeframe='1h', limit=24)
            if oi_hist and len(oi_hist) >= 2:
                first_oi = oi_hist[0].get('openInterestValue') or oi_hist[0].get('openInterestAmount')
                last_oi = oi_hist[-1].get('openInterestValue') or oi_hist[-1].get('openInterestAmount')
                if first_oi and float(first_oi) > 0 and last_oi is not None:
                    oi_change_24h_value = (float(last_oi) - float(first_oi)) / float(first_oi) * 100
                    oi_change_24h = self._format_pct(oi_change_24h_value, digits=2)
        except Exception:
            pass

        liquidation_risk_level = LiquidationRiskLevel.LOW
        risk_note = "资金费率与持仓变化平稳，未见明显清算拥挤风险"

        funding_abs_pct = abs(funding_rate * 100)
        oi_change_abs = abs(oi_change_24h_value) if oi_change_24h_value is not None else 0.0
        price_change_abs = abs(price_change_24h_pct) if price_change_24h_pct is not None else 0.0

        if (funding_abs_pct > 0.03 and oi_change_abs > 10) or (oi_change_abs > 15 and price_change_abs > 4):
            liquidation_risk_level = LiquidationRiskLevel.HIGH
            risk_note = "资金费率偏极端且持仓变化剧烈，存在拥挤方向连环清算风险"
        elif (funding_abs_pct > 0.015 and oi_change_abs > 6) or (oi_change_abs > 8 and price_change_abs > 2.5):
            liquidation_risk_level = LiquidationRiskLevel.MEDIUM
            risk_note = "资金费率或持仓变化偏热，需防插针扫损与短时清算波动"

        return DerivativesContext(
            funding_rate=round(funding_rate, 8),
            funding_rate_pct=funding_rate_pct,
            next_funding_time=next_funding_time,
            open_interest_value=round(open_interest_value, 4),
            open_interest_change_24h=oi_change_24h,
            liquidation_risk_level=liquidation_risk_level,
            liquidation_risk_note=risk_note,
        )

    def generate_input(self, symbol: str, request_id: str, learning_bridge: LearningBridge) -> TechnicalAnalystInput:
        """
        全自动生成技术分析师的输入数据
        """
        # 1. 抓取各周期数据（日内交易优化）
        df_m5 = self.fetch_ohlcv_df(symbol, '5m', limit=100)    # ~8小时
        df_m15 = self.fetch_ohlcv_df(symbol, '15m', limit=200)  # ~50小时，足够算指标
        df_h1 = self.fetch_ohlcv_df(symbol, '1h', limit=100)    # ~4天
        df_h4 = self.fetch_ohlcv_df(symbol, '4h', limit=100)    # ~17天

        # 2. 构建 MarketSnapshot
        current_price = df_m5.iloc[-1]['close']

        # 24h 涨跌幅：从交易所 ticker 获取精确数据
        ticker_pct: Optional[float] = None
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            pct = ticker.get('percentage')  # 交易所返回的精确 24h 涨跌幅
            if pct is not None:
                ticker_pct = float(pct)
                price_change_24h = f"{'+' if pct >= 0 else ''}{pct:.2f}%"
            else:
                price_change_24h = "N/A"
        except Exception as e:
            # logging.warning(f"fetch_ticker failed: {e}, falling back to D1 calc")
            price_change_24h = "N/A"

        # M5: 最近 12 根（1小时），同时填充 bar_status + vol_ratio
        avg_vol_m5 = df_m5['volume'].tail(20).mean()
        m5_candles = []
        for _, row in df_m5.tail(12).iterrows():
            m5_candles.append(Candle(
                t=row['datetime'].strftime('%H:%M'),
                o=row['open'], h=row['high'], l=row['low'], c=row['close'], v=row['volume'],
                bar_status=self.get_bar_status(row['open'], row['high'], row['low'], row['close']),
                vol_ratio=round(row['volume'] / avg_vol_m5, 2) if avg_vol_m5 > 0 else None
            ))

        # M15: 最近 20 根（5小时），同时填充 bar_status + vol_ratio
        avg_vol_m15 = df_m15['volume'].tail(20).mean()
        m15_candles = []
        for _, row in df_m15.tail(20).iterrows():
            m15_candles.append(Candle(
                t=row['datetime'].strftime('%H:%M'),
                o=row['open'], h=row['high'], l=row['low'], c=row['close'], v=row['volume'],
                bar_status=self.get_bar_status(row['open'], row['high'], row['low'], row['close']),
                vol_ratio=round(row['volume'] / avg_vol_m15, 2) if avg_vol_m15 > 0 else None
            ))

        # H1: 最近 24 根（1天），日内趋势确认
        avg_vol_h1 = df_h1['volume'].tail(24).mean()
        h1_candles = []
        for _, row in df_h1.tail(24).iterrows():
            h1_candles.append(Candle(
                t=row['datetime'].strftime('%m-%d %H:%M'),
                o=row['open'], h=row['high'], l=row['low'], c=row['close'], v=row['volume'],
                bar_status=self.get_bar_status(row['open'], row['high'], row['low'], row['close']),
                vol_ratio=round(row['volume'] / avg_vol_h1, 2) if avg_vol_h1 > 0 else None
            ))

        snapshot = MarketSnapshot(
            current_price=current_price,
            price_change_24h=price_change_24h,
            m5_candles=m5_candles,
            m15_candles=m15_candles,
            h1_candles=h1_candles,
        )

        # 3. 计算多维指标层
        m15_analysis = self.calculate_indicators_m15(df_m15)
        htf_context = self.get_htf_context(df_h4, df_h1)
        derivatives_context = self.get_derivatives_context(symbol, ticker_pct)
        indicators = TechnicalIndicators(
            m15_1000_bars=m15_analysis,
            htf_context=htf_context,
            derivatives_context=derivatives_context,
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
            # learning_bridge=learning_bridge
        )
