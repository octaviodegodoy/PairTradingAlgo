import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from constants import TRADING_PAIR_X, TRADING_PAIR_Y, ROLLING_PERIODS
from mt5_connector import MT5Connector


# ============================================================
# Filtro de Kalman de 2ª Ordem para Pair Trading
# ============================================================

class KalmanPairTrading2ndOrder:
    """
    Filtro de Kalman de 2ª ordem aplicado a Pair Trading.

    Estado: [alpha, beta, d_beta, dd_beta]
        - alpha   : intercepto da regressão
        - beta    : hedge ratio
        - d_beta  : velocidade de variação do beta
        - dd_beta : aceleração de variação do beta

    Observação:
        y_t = alpha + beta * x_t + ruído

    The ``filter_batch`` method mirrors the interface of ``KalmanFilter`` so
    that ``get_dynamic_spread_zscores`` in utils.py can use either order
    transparently via the ``KALMAN_ORDER`` constant.
    """

    def __init__(self, delta=1e-4, sigma_obs=1.0, dt=1.0):
        self.dt = dt
        self.n_states = 4

        self.x = np.zeros((self.n_states, 1))

        self.F = np.array([
            [1, 0,       0,             0],
            [0, 1,      dt, 0.5 * dt**2],
            [0, 0,       1,           dt],
            [0, 0,       0,            1]
        ])

        q_alpha = delta * 0.1
        q_beta = delta
        q_dbeta = delta * 2
        q_ddbeta = delta * 4
        self.Q = np.diag([q_alpha, q_beta, q_dbeta, q_ddbeta])

        self.R = np.array([[sigma_obs**2]])
        self.P = np.eye(self.n_states) * 1.0

        self.history = {
            "alpha": [], "beta": [], "d_beta": [], "dd_beta": [],
            "spread": [], "spread_std": [], "log_likelihood": []
        }

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, y, x_asset):
        H = np.array([[1, x_asset, 0, 0]])
        y_pred = H @ self.x
        innovation = np.array([[y]]) - y_pred
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation

        I = np.eye(self.n_states)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T

        log_lik = -0.5 * (np.log(2 * np.pi * S[0, 0]) +
                          innovation[0, 0]**2 / S[0, 0])

        spread = innovation[0, 0]
        spread_std = np.sqrt(S[0, 0])

        self.history["alpha"].append(self.x[0, 0])
        self.history["beta"].append(self.x[1, 0])
        self.history["d_beta"].append(self.x[2, 0])
        self.history["dd_beta"].append(self.x[3, 0])
        self.history["spread"].append(spread)
        self.history["spread_std"].append(spread_std)
        self.history["log_likelihood"].append(log_lik)

        return spread, spread_std

    def step(self, y, x_asset):
        self.predict()
        return self.update(y, x_asset)

    def get_state(self):
        return {
            "alpha": self.x[0, 0],
            "beta": self.x[1, 0],
            "d_beta": self.x[2, 0],
            "dd_beta": self.x[3, 0]
        }

    # ------------------------------------------------------------------
    # Unified interface — matches KalmanFilter.filter_batch signature so
    # get_dynamic_spread_zscores can use either order transparently.
    # ------------------------------------------------------------------

    def filter_batch(self, y: np.ndarray, x: np.ndarray) -> pd.DataFrame:
        """
        Run the 2nd-order Kalman filter on a batch of log-price data.

        Parameters
        ----------
        y : np.ndarray  — log-prices of the dependent asset
        x : np.ndarray  — log-prices of the independent asset

        Returns
        -------
        pd.DataFrame with columns matching KalmanFilter.filter_batch:
            ``y``, ``x``, ``kalman_hedge_ratio``, ``spread``,
            ``spread_mean``, ``spread_std``, ``zscore``
        """
        assert len(y) == len(x), "y and x must have the same length"

        # Reset history for a fresh batch run
        self.x = np.zeros((self.n_states, 1))
        self.P = np.eye(self.n_states) * 1.0
        self.history = {
            "alpha": [], "beta": [], "d_beta": [], "dd_beta": [],
            "spread": [], "spread_std": [], "log_likelihood": []
        }

        for i in range(len(y)):
            self.step(y[i], x[i])

        hedge_ratios = np.array(self.history["beta"])
        # The spread stored in history is the *innovation* (y - H*x_pred).
        # For consistency with KalmanFilter we recompute the residual spread
        # as  y - alpha - beta * x  using the posterior state estimates.
        alphas = np.array(self.history["alpha"])
        spreads = y - alphas - hedge_ratios * x

        results = pd.DataFrame({
            "y": y,
            "x": x,
            "kalman_hedge_ratio": hedge_ratios,
            "spread": spreads,
        })

        results["spread_mean"] = results["spread"].rolling(window=ROLLING_PERIODS).mean()
        results["spread_std"] = results["spread"].rolling(window=ROLLING_PERIODS).std()
        results["zscore"] = (results["spread"] - results["spread_mean"]) / results["spread_std"]

        return results


# ============================================================
# Visualização — Apenas Z-Score
# ============================================================

def plot_zscore(dates, kf, symbol_x, symbol_y, entry_z=2.0, exit_z=0.5):
    spreads = np.array(kf.history["spread"])
    spread_stds = np.array(kf.history["spread_std"])
    z_score = spreads / np.where(spread_stds > 1e-8, spread_stds, 1)

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(dates, z_score, color="purple", linewidth=0.9, alpha=0.9, label="Z-Score")

    ax.axhline(y=entry_z, color="red", linestyle="--", linewidth=1, label=f"Entrada (±{entry_z}σ)")
    ax.axhline(y=-entry_z, color="red", linestyle="--", linewidth=1)
    ax.axhline(y=exit_z, color="green", linestyle=":", linewidth=1, label=f"Saída (±{exit_z}σ)")
    ax.axhline(y=-exit_z, color="green", linestyle=":", linewidth=1)
    ax.axhline(y=0, color="k", linewidth=0.5)

    ax.fill_between(dates, -entry_z, entry_z, alpha=0.05, color="red")
    ax.fill_between(dates, z_score, entry_z,
                    where=z_score > entry_z, color="red", alpha=0.3, label="Short spread")
    ax.fill_between(dates, z_score, -entry_z,
                    where=z_score < -entry_z, color="green", alpha=0.3, label="Long spread")

    ax.set_title(f"Z-Score do Spread — {symbol_y} ~ α + β·{symbol_x}  (Kalman 2ª Ordem)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Data")
    ax.set_ylabel("Z-Score")
    ax.set_ylim(-5, 5)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig("kalman_zscore.png", dpi=150, bbox_inches="tight")
    plt.show()


# ============================================================
# Adaptador genérico para extrair preços de get_data()
# ============================================================

def _extract_prices(data_x, data_y, symbol_x, symbol_y):
    if hasattr(data_x, "columns") and hasattr(data_x, "index"):
        import pandas as pd
        col_x = _find_price_column(data_x)
        col_y = _find_price_column(data_y)
        df = pd.DataFrame({
            symbol_x: data_x[col_x],
            symbol_y: data_y[col_y]
        }).dropna()
        return df[symbol_x].values.astype(float), df[symbol_y].values.astype(float), df.index.values

    if isinstance(data_x, dict):
        price_key = _find_dict_key(data_x, ["prices", "price", "close", "Close", "values"])
        date_key = _find_dict_key(data_x, ["dates", "date", "index", "timestamp", "time"])
        px = np.array(data_x[price_key], dtype=float)
        py = np.array(data_y[price_key], dtype=float)
        if date_key:
            dx = np.array(data_x[date_key])
            dy = np.array(data_y[date_key])
            common = np.intersect1d(dx, dy)
            mask_x = np.isin(dx, common)
            mask_y = np.isin(dy, common)
            return px[mask_x], py[mask_y], common
        else:
            n = min(len(px), len(py))
            return px[:n], py[:n], np.arange(n)

    if isinstance(data_x, (list, np.ndarray)):
        px = np.array(data_x, dtype=float)
        py = np.array(data_y, dtype=float)
        n = min(len(px), len(py))
        return px[:n], py[:n], np.arange(n)

    raise ValueError(
        f"Formato de get_data() não reconhecido: {type(data_x)}. "
        f"Adapte a função _extract_prices() para seu formato."
    )


def _find_price_column(df):
    candidates = ["Close", "close", "Adj Close", "adj_close", "price", "Price", "Last"]
    for col in candidates:
        if col in df.columns:
            return col
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        return numeric_cols[-1]
    raise ValueError(f"Coluna de preço não encontrada. Colunas: {list(df.columns)}")


def _find_dict_key(d, candidates):
    for key in candidates:
        if key in d:
            return key
    return None


# ============================================================
# Main
# ============================================================

def main():
    # ---------------------------------------------------------
    # CONFIGURAÇÃO
    # ---------------------------------------------------------
    mt5_conn = MT5Connector()

    DELTA = 1e-4
    SIGMA_OBS = 0.01
    ENTRY_Z = 2.0
    EXIT_Z = 0.5

    # ---------------------------------------------------------
    # 1. Obter dados via get_data()
    # ---------------------------------------------------------
    print(f"📥 Obtendo dados de {TRADING_PAIR_X[0]}...")
    data_x = mt5_conn.get_data_futures_btg(TRADING_PAIR_X[0])
 

    print(f"📥 Obtendo dados de {TRADING_PAIR_Y[0]}...")
    data_y = mt5_conn.get_data_futures_btg(TRADING_PAIR_Y[0])

    # ---------------------------------------------------------
    # 2. Extrair e alinhar preços
    # ---------------------------------------------------------
    prices_x, prices_y, dates = _extract_prices(data_x, data_y, TRADING_PAIR_X[0], TRADING_PAIR_Y[0])
    print(f"✅ {len(prices_x)} observações alinhadas.")

    # ---------------------------------------------------------
    # 3. Log-preços
    # ---------------------------------------------------------
    log_x = np.log(prices_x)
    log_y = np.log(prices_y)

    # ---------------------------------------------------------
    # 4. Filtro de Kalman 2ª Ordem
    # ---------------------------------------------------------
    print("📊 Aplicando Filtro de Kalman de 2ª Ordem...")
    kf = KalmanPairTrading2ndOrder(delta=DELTA, sigma_obs=SIGMA_OBS, dt=1.0)

    for i in range(len(log_x)):
        kf.step(y=log_y[i], x_asset=log_x[i])

    # ---------------------------------------------------------
    # 5. Plotar apenas o Z-Score
    # ---------------------------------------------------------
    print("📊 Gerando gráfico de Z-Score...")
    plot_zscore(dates, kf, TRADING_PAIR_X[0], TRADING_PAIR_Y[0], entry_z=ENTRY_Z, exit_z=EXIT_Z)

    print("✅ Concluído! Gráfico salvo em 'kalman_zscore.png'")


if __name__ == "__main__":
    main()