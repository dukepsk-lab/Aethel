from aethel.config import get_settings
from aethel.mt5.base import MT5Client


def get_mt5_client() -> MT5Client:
    s = get_settings()
    if s.mt5_mode == "direct":
        from aethel.mt5.direct import DirectMT5Client

        return DirectMT5Client()
    from aethel.mt5.gateway import GatewayMT5Client

    return GatewayMT5Client(s.mt5_gateway_url, s.mt5_gateway_token)
