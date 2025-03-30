# modules/backtester.py
"""バックテスト(シミュレーション)関連のモジュール

       * 学習済みモデルの推定結果(上昇/下降確率など)を用い、
         スプレッドを考慮した売買シミュレーションを行う。
       * ただし「ロングを決済してすぐまたロング」を実行しようとした場合は
         実際にはポジションを継続し、スプレッドコストも発生しないようにする。

"""

import numpy as np
from tqdm import tqdm


class Backtester:
    """バックテストのロジックを実装するクラス

    予測結果(上昇確率)に基づいて、買い/売り/ノーポジを切り替えるシンプル戦略をシミュレーション。
    スプレッドを考慮し、連続する同方向ポジションは
    「実際には一度決済せずに継続している」とみなしてコストを発生させない。

    Attributes:
        price_data (numpy.ndarray): shape=(N,) の生価格データ
        hold_term (int): 何ステップ保有するかの最大値
        threshold_long (float): 上昇確率がこの値以上なら買い
        threshold_short (float): 上昇確率がこの値以下なら売り
        risk_cut (float): 損切りライン(±%)
        take_profit (float): 利確ライン(±%)
        spread (float): スプレッド(手数料率)。例: 0.001 → 0.1%

    """

    def __init__(self,
                 price_data: np.ndarray,
                 hold_term: int = 30,
                 threshold_long: float = 0.55,
                 threshold_short: float = 0.45,
                 risk_cut: float = 0.05,
                 take_profit: float = 0.05,
                 spread: float = 0.0005):
        """
        Args:
            price_data (np.ndarray): shape=(N,) の生価格
            hold_term (int): ポジションを保有し続ける最大ステップ数
            threshold_long (float): 上昇確率がこれ以上なら買い
            threshold_short (float): 上昇確率がこれ以下なら売り
            risk_cut (float): 損切りライン(±%)
            take_profit (float): 利確ライン(±%)
            spread (float): スプレッド(0.001→0.1%など)
        """
        self.price_data = price_data
        self.hold_term = hold_term
        self.threshold_long = threshold_long
        self.threshold_short = threshold_short
        self.risk_cut = risk_cut
        self.take_profit = take_profit
        self.spread = spread

        self.step_logs = []  # (step, price, up_prob, position, cumulative_pnl)

    def run_backtest(self, up_prob_array: np.ndarray, show_progress=False):
        """バックテストを実行する

        1. ノーポジ時に「上昇確率 >= threshold_long」ならロング、「<= threshold_short」ならショートを建てる
           → 建てるときにスプレッド(1回)を引く
        2. ポジション保有中に
           - 保有期間切れ
           - 損切りライン到達
           - 利確ライン到達
           となれば決済しようとする
           ただし「次ステップですぐ同方向をエントリーするなら決済をスキップ(=継続)」する
           → 本当に決済するときだけスプレッド(1回)を引き、PnLを確定
        3. ショート→ロングなど逆方向へのフリップは、「決済→エントリー」で 2回分スプレッドがかかる
           ただし 直後に同方向になるケースはスキップロジックによって実質フリップしない (連続ホールド)

        Args:
            up_prob_array (np.ndarray): shape=(N,)のUP確率
            show_progress (bool): tqdmで進捗バーを表示するか

        Returns:
            tuple (final_pnl, pnl_history, win_count, lose_count):
                final_pnl (float): 最終的な累積損益
                pnl_history (list of float): ステップごとの累積損益
                win_count (int): 勝ちトレード数
                lose_count (int): 負けトレード数
        """
        num_steps = len(self.price_data)
        iterator = range(num_steps)
        if show_progress:
            iterator = tqdm(iterator)

        position = 0  # 1: ロング, -1: ショート, 0: ノーポジ
        entry_price = 0.0
        entry_index = 0

        cumulative_pnl = 0.0
        pnl_history = []
        win_count = 0
        lose_count = 0

        for i in iterator:
            current_price = self.price_data[i]
            up_prob = up_prob_array[i]

            # --------------------------------
            # 1) 保有中の場合 -> 損益判定
            # --------------------------------
            if position != 0:
                change_ratio = (current_price - entry_price) / entry_price
                if position == -1:
                    change_ratio = -change_ratio  # ショートは反転

                # (a) 保有期間終了
                hold_duration = i - entry_index
                want_to_close = (hold_duration >= self.hold_term)

                # (b) 利確 or 損切り
                if change_ratio >= self.take_profit:
                    want_to_close = True
                elif change_ratio <= -self.risk_cut:
                    want_to_close = True

                if want_to_close:
                    # ---- 次ステップで同じ方向をまたエントリーするか？ => その場合は決済スキップ(=継続) ----
                    # ただし i == num_steps-1 ならもう次ステップなし
                    if i < num_steps - 1:
                        next_up_prob = up_prob_array[i + 1]
                        # 次ステップがロングと判断されるなら position=1、ショートなら-1、ノーポジなら0
                        next_position = 0
                        if next_up_prob >= self.threshold_long:
                            next_position = 1
                        elif next_up_prob <= self.threshold_short:
                            next_position = -1

                        # 「今のポジション」と「次ステップのポジション」が同じ方向なら
                        # 今回の決済はスキップしてポジション継続
                        if next_position == position:
                            # skip close
                            pass
                        else:
                            # 決済実行
                            trade_pnl = (
                                current_price - entry_price) if position == 1 else (entry_price - current_price)
                            # スプレッドを1回差し引く
                            spread_cost = entry_price * self.spread  # entry_price基準
                            trade_pnl -= spread_cost

                            cumulative_pnl += trade_pnl
                            if trade_pnl > 0:
                                win_count += 1
                            else:
                                lose_count += 1
                            position = 0

                    else:
                        # 最終ステップなら無条件で決済
                        trade_pnl = (
                            current_price - entry_price) if position == 1 else (entry_price - current_price)
                        spread_cost = entry_price * self.spread
                        trade_pnl -= spread_cost

                        cumulative_pnl += trade_pnl
                        if trade_pnl > 0:
                            win_count += 1
                        else:
                            lose_count += 1
                        position = 0

            # --------------------------------
            # 2) ノーポジの場合 -> エントリー判定
            # --------------------------------
            if position == 0:
                if up_prob >= self.threshold_long:
                    # ロングエントリー
                    position = 1
                    entry_price = current_price
                    entry_index = i
                    # スプレッド分コストを差し引く (エントリー時に払う想定)
                    # entry_price基準で差し引く
                    cumulative_pnl -= (entry_price * self.spread)

                elif up_prob <= self.threshold_short:
                    # ショートエントリー
                    position = -1
                    entry_price = current_price
                    entry_index = i
                    cumulative_pnl -= (entry_price * self.spread)

            pnl_history.append(cumulative_pnl)
            self.step_logs.append(
                (i, current_price, up_prob, position, cumulative_pnl))

        return cumulative_pnl, pnl_history, win_count, lose_count

    def get_step_logs(self):
        """ステップごとのログ(step, price, up_prob, position, cumulative_pnl)を返す"""
        return self.step_logs
