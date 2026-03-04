from typing import Dict, List
from fastapi import WebSocket, WebSocketDisconnect
import logging

logger = logging.getLogger(__name__)


class PriceStreamManager:
    """Manage WebSocket connections for live price updates."""
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}  # {symbol: [ws, ...]}

    async def connect(self, symbol: str, websocket: WebSocket):
        await websocket.accept()
        if symbol not in self.active_connections:
            self.active_connections[symbol] = []
        self.active_connections[symbol].append(websocket)
        logger.info(f"[WS] {symbol} connected, total: {len(self.active_connections[symbol])}")

    async def disconnect(self, symbol: str, websocket: WebSocket):
        if symbol in self.active_connections:
            self.active_connections[symbol].remove(websocket)
            if not self.active_connections[symbol]:
                del self.active_connections[symbol]
        logger.info(f"[WS] {symbol} disconnected")

    async def broadcast(self, symbol: str, message: dict):
        """Send update to all clients listening to this symbol."""
        if symbol not in self.active_connections:
            return
        disconnected = []
        for ws in self.active_connections[symbol]:
            try:
                await ws.send_json(message)
            except Exception as e:
                logger.debug(f"[WS] send error for {symbol}: {e}")
                disconnected.append(ws)
        # cleanup dead connections
        for ws in disconnected:
            await self.disconnect(symbol, ws)


manager = PriceStreamManager()
