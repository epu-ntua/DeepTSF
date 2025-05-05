from typing import List
import httpx
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from .config import settings

async def validate_remotely(token, issuer, client_id, client_secret):
    headers = {
        'accept': 'application/json',
        'cache-control': 'no-cache',
        'content-type': 'application/x-www-form-urlencoded',
    }
    data = {
        'client_id': client_id,
        'client_secret': client_secret,
        'token': token,
    }
    url = issuer + '/introspect'

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, data=data)
    
    return response

async def validate_websocket_token(token: str) -> List[str]:
    """Validate the token for WebSocket connection and return user roles if valid."""
    res = await validate_remotely(
        token=token,
        issuer=settings.token_issuer,
        client_id=settings.client_id,
        client_secret=settings.client_secret,
    )
    if res.status_code == httpx.codes.OK and res.json()['active']:
        return res.json().get('realm_access', {}).get('roles', [])
    else:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

class WebSocketAuthValidator:
    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = allowed_roles

    async def __call__(self, websocket: WebSocket) -> bool:
        # Extract token from query parameters
        token = websocket.query_params.get("token")

        print(token, "HERE")
        
        if not token:
            print(token, "HERESDDSSD")
            await websocket.close(code=1008)
            raise WebSocketDisconnect(code=1008)

        try:
            # Validate token and retrieve roles
            roles = await validate_websocket_token(token)
            if any(role in self.allowed_roles for role in roles):
                return True
            else:
                await websocket.close(code=1008)
                raise WebSocketDisconnect(code=1008)
        except HTTPException:
            await websocket.close(code=1008)
            raise WebSocketDisconnect(code=1008)

# Create role-specific WebSocket validators
websocket_admin_validator = WebSocketAuthValidator(allowed_roles=settings.admin_routes_roles)
websocket_scientist_validator = WebSocketAuthValidator(allowed_roles=settings.scientist_routes_roles)
websocket_engineer_validator = WebSocketAuthValidator(allowed_roles=settings.engineer_routes_roles)
websocket_common_validator = WebSocketAuthValidator(allowed_roles=settings.common_routes_roles)