import re
from typing import Dict, Any, Optional


class StrategyParser:
    """
    Parses natural language trading rules into structured backtesting parameters.
    Supports:
      - Single ticker rules (e.g., "Buy when AAPL drops 5% in three days")
      - Stop-loss and take-profit
      - Sector-wide rules ("using all tickers in technology sector")
      - Rolling window references ("5-day high", "10 day average")
      - Capital allocation phrases ("buy with 50% of capital")
    """

    ACTIONS = {"buy", "sell", "short", "cover"}

    TIME_UNITS = {
        "day": 1, "days": 1,
        "week": 7, "weeks": 7,
        "month": 30, "months": 30
    }

    NUMBER_WORDS = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "twenty": 20, "thirty": 30
    }

    def parse(
        self,
        text: str,
        symbol_map: Optional[Dict[str, str]] = None,
        known_symbols: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Main NLP entrypoint.
        Detects sector-wide phrasing, entry/exit rules, stop-loss, take-profit, and allocation.
        """
        text = text.strip().lower()

        # Detect sector
        sector_match = re.search(
            r"(?:using|for|in)\s+(?:all\s+tickers\s+in|the)?\s*([\w\s]+?)\s+(?:sector|industry)", text)
        sector_name = None
        if sector_match:
            sector_name = sector_match.group(1).strip()
            text = re.sub(
                r"(?:using|for|in)\s+(?:all\s+tickers\s+in|the)?\s*[\w\s]+?\s+(?:sector|industry)", "", text)

        # Split entry/exit
        parts = re.split(r"\b(?:and then|then|and after|and)\b", text)
        entry_text = parts[0].strip()
        exit_text = parts[1].strip() if len(parts) > 1 else None

        entry = self._parse_single(entry_text, symbol_map, known_symbols)
        exit_rule = self._parse_single(exit_text, symbol_map, known_symbols) if exit_text else None

        # Extract stoploss, takeprofit, allocation globally
        stop_loss, take_profit = self._extract_stop_take(text)
        allocation = self._extract_allocation(text)

        # Add to exit rule if not already
        if exit_rule:
            if stop_loss and not exit_rule["condition"].get("stop_loss"):
                exit_rule["condition"]["stop_loss"] = stop_loss
            if take_profit and not exit_rule["condition"].get("take_profit"):
                exit_rule["condition"]["take_profit"] = take_profit

        result = {"entry": entry, "exit": exit_rule}
        if sector_name:
            result["sector"] = sector_name
        if allocation:
            result["allocation"] = allocation
        if stop_loss:
            result["stop_loss"] = stop_loss
        if take_profit:
            result["take_profit"] = take_profit

        return result

    # ----------------------------------------------------------------------
    # Parse a single rule
    # ----------------------------------------------------------------------
    def _parse_single(
        self,
        text: Optional[str],
        symbol_map: Optional[Dict[str, str]],
        known_symbols: Optional[list]
    ) -> Optional[Dict[str, Any]]:
        if not text:
            return None

        original_text = text
        text = text.lower()

        result = {
            "action": None,
            "symbol": None,
            "condition": {
                "metric": None,
                "direction": None,
                "threshold": None,
                "period_days": None
            }
        }

        # Action
        for a in self.ACTIONS:
            if re.search(rf"\b{a}\b", text):
                result["action"] = a
                break

        # Symbol
        symbol = self._extract_symbol(original_text)
        if not symbol:
            symbol = self._extract_symbol_from_known(text, known_symbols)
        if not symbol and symbol_map:
            for name, ticker in symbol_map.items():
                if name.lower() in text:
                    symbol = ticker
                    break
        result["symbol"] = symbol

        # Direction
        if re.search(r"\b(drop|drops|down|fall|falls|decrease|plunge)\b", text):
            result["condition"]["direction"] = "down"
        elif re.search(r"\b(rise|rises|up|gain|gains|increase)\b", text):
            result["condition"]["direction"] = "up"

        # Threshold (%)
        m_pct = re.search(
            r"(\d+(?:\.\d+)?)\s*%|\b(\d+(?:\.\d+)?)\s*(percent|percentage)\b", text)
        if m_pct:
            val = m_pct.group(1) or m_pct.group(2)
            result["condition"]["threshold"] = float(val) / 100.0

        # Period (days/weeks)
        m_period = re.search(
            r"\b(in|within|over|from|past)\s+(?P<num>\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|twenty|thirty)\s*(?P<unit>day|days|week|weeks|month|months)\b",
            text
        )
        if not m_period:
            m_period = re.search(
                r"(?P<num>\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|twenty|thirty)[-\s]*(?P<unit>day|days|week|weeks|month|months)",
                text
            )
        if m_period:
            num_raw = m_period.group("num")
            unit = m_period.group("unit")
            num = int(num_raw) if num_raw.isdigit() else self.NUMBER_WORDS.get(num_raw, None)
            if num is not None:
                result["condition"]["period_days"] = num * self.TIME_UNITS[unit]

        if result["condition"]["threshold"] is not None:
            result["condition"]["metric"] = "price_change"

        return result

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def _extract_symbol(self, original_text: str) -> Optional[str]:
        tokens = re.findall(r"\b[A-Z]{1,6}\b", original_text)
        for token in tokens:
            if token.lower() in self.ACTIONS:
                continue
            if len(token) >= 2:
                return token
        return None

    def _extract_symbol_from_known(self, lower_text: str, known_symbols: Optional[list]) -> Optional[str]:
        if not known_symbols:
            return None
        for sym in known_symbols:
            s = sym.lower()
            if re.search(rf"\b{re.escape(s)}\b", lower_text):
                return sym.upper()
        return None

    def _extract_stop_take(self, text: str):
        stop_loss = None
        take_profit = None
        m_stop = re.search(r"(?:stop\s*loss|stoploss|falls?\s+another)\s*(\d+(?:\.\d+)?)\s*%", text)
        m_take = re.search(r"(?:take\s*profit|tp|rises?\s+another)\s*(\d+(?:\.\d+)?)\s*%", text)
        if m_stop:
            stop_loss = float(m_stop.group(1)) / 100.0
        if m_take:
            take_profit = float(m_take.group(1)) / 100.0
        return stop_loss, take_profit

    def _extract_allocation(self, text: str) -> Optional[float]:
        match = re.search(r"(\d+(?:\.\d+)?)\s*%?\s*(?:percent)?\s*(?:of\s+)?(?:the\s+)?capital", text)
        if match:
            val = float(match.group(1))
            if val > 1:
                val = val / 100.0
            return val
        return None
