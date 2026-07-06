"""バックテスト全体のパラメータ定義。

数値はすべて「一般的な目安」であり、最適化の起点。
変更したい場合はこのファイルを直接編集するか、main.py の引数で上書きする。
"""
from dataclasses import dataclass


@dataclass
class Config:
    # ---- 期間・ユニバース ----
    start: str = "2023-01-01"        # バックテスト開始日
    end: str = ""                    # 空なら取得できる最新日まで
    universe_size: int = 200         # 売買代金上位N銘柄(プライム市場)
    liquidity_lookback_days: int = 60  # ユニバース選定に使う直近営業日数
    warmup_days: int = 80            # 指標計算用に start より前に余分に取る日数

    # ---- 資金管理 ----
    initial_equity: float = 3_000_000
    risk_per_trade: float = 0.01     # 1トレードの損失許容 = 資金の1%
    max_positions: int = 5
    max_position_weight: float = 0.30  # 1銘柄の最大投入比率
    monthly_dd_stop: float = -0.05   # 月間DDがこれを下回ったら当月の新規建て停止
    commission_rate: float = 0.0005  # 片道手数料
    slippage_rate: float = 0.001     # 片道スリッページ
    unit: int = 100                  # 単元株数
    day_trade_only: bool = False     # Trueなら持ち越し禁止(当日引けで全決済)

    # ---- ① 地合いレジーム判定 ----
    regime_sma: int = 25
    regime_slope_days: int = 5       # 25日線の傾き判定に使う日数
    regime_adx_period: int = 14
    regime_adx_low: float = 20.0     # これ未満なら方向感なし→サイズ半分

    # ---- ② 順張りスイング(モメンタム押し目買い) ----
    mom_rs_lookback: int = 63        # レラティブストレングス計算期間(約3ヶ月)
    mom_rs_top_pct: float = 0.10     # RS上位10%のみ対象
    mom_pullback_min: float = 0.05   # 高値からの押し幅 下限
    mom_pullback_max: float = 0.12   # 高値からの押し幅 上限
    mom_days_since_high: tuple = (3, 7)  # 高値からの経過日数
    mom_stop_max: float = 0.06       # 損切り幅がこれを超える場合は見送り
    mom_tp_r_multiple: float = 2.0   # リスクの2倍で半分利確
    mom_trail_sma: int = 10          # 終値がこのMAを割れたら翌日寄り手仕舞い
    mom_max_hold: int = 15           # 最大保有日数

    # ---- ③ 逆張りスイング(乖離率 + RSI(2)) ----
    mr_dev25_threshold: float = -0.15  # 25日線乖離率
    mr_rsi2_threshold: float = 10.0
    mr_volume_mult: float = 2.0      # 出来高が20日平均の2倍以上
    mr_stop: float = 0.07            # 損切り -7%
    mr_time_stop: int = 5            # 5日たって含み損なら撤退
    mr_max_hold: int = 10

    # ---- ④ 決算PEAD ----
    pead_gap_min: float = 0.05       # 決算翌日のギャップ +5%以上
    pead_close_range_pct: float = 0.7  # 終値が当日レンジの上位30%
    pead_entry_from: int = 2         # 反応日から2〜5日目の押し目で入る
    pead_entry_to: int = 5
    pead_pullback: float = 0.03      # 反応日終値から3%以上の押し
    pead_max_hold: int = 20

    # ---- 仕手株・低流動性の除外 ----
    mr_min_price: float = 300.0      # 逆張りは株価がこれ未満の低位株を除外
    mr_scale_categories: tuple = (   # 逆張りはTOPIX規模区分がこれらの銘柄のみ
        "TOPIX Core30", "TOPIX Large70", "TOPIX Mid400",
    )
    participation_cap: float = 0.05  # 建玉上限 = 20日平均出来高の5%
    exclude_margin_alert: bool = True  # 日々公表(信用規制)銘柄をエントリー禁止
    margin_alert_lookback: int = 10  # 直近この営業日数内に日々公表掲載があれば除外

    # ---- ⑤⑥ 共通フィルター ----
    margin_ratio_max: float = 5.0    # 信用倍率がこれ超なら新規買い見送り
    margin_publish_lag: int = 3      # 信用残の公表ラグ(営業日)
    earnings_avoid_days: int = 2     # 決算発表の2営業日前までに手仕舞い
    pead_earnings_avoid_days: int = 5  # PEADは次回決算の5営業日前まで
    sq_size_mult: float = 0.5        # メジャーSQ週はサイズ半分
    adx_low_size_mult: float = 0.5   # ADX低迷時はサイズ半分
