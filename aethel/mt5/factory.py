from aethel.config import get_settings
from aethel.mt5.base import MT5Client

_client: MT5Client | None = None


def get_mt5_client() -> MT5Client:
    global _client
    if _client is not None:
        return _client
    s = get_settings()
    if s.mt5_mode == "direct":
        from aethel.mt5.direct import DirectMT5Client

        _client = DirectMT5Client()
    else:
        from aethel.mt5.gateway import GatewayMT5Client

        _client = GatewayMT5Client(s.mt5_gateway_url, s.mt5_gateway_token)
    return _client
