"""Internal multi-image API, started and stopped by the evaluation runner."""
from __future__ import annotations

import base64
import binascii
import io
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from .qwen3vl_service import Qwen3VLService, ServiceConfig


class ChatRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]] = Field(min_length=1)
    max_tokens: int = Field(default=256, gt=0)
    temperature: float = Field(default=0, ge=0)
    stream: bool = False


def resolve_messages(messages):
    resolved, images = [], []
    for message in messages:
        role = message.get('role')
        if role not in ('system', 'user', 'assistant'):
            raise HTTPException(400, 'Unsupported message role')
        content = message.get('content')
        if isinstance(content, str):
            content = [{'type': 'text', 'text': content}]
        if not isinstance(content, list) or not content:
            raise HTTPException(400, 'Message content must be nonempty')
        parts = []
        for part in content:
            if not isinstance(part, dict):
                raise HTTPException(400, 'Invalid message content')
            if part.get('type') == 'text' and isinstance(part.get('text'), str):
                parts.append({'type': 'text', 'text': part['text']})
            elif part.get('type') == 'image_url' and role == 'user':
                source = part.get('image_url', {})
                url = source.get('url', '') if isinstance(source, dict) else ''
                if not isinstance(url, str) or not url.startswith('data:image/'):
                    raise HTTPException(400, 'Evaluation images must be base64 data URLs')
                header, _, data = url.partition(',')
                if not header.endswith(';base64'):
                    raise HTTPException(400, 'Expected a base64 image')
                try:
                    with Image.open(io.BytesIO(base64.b64decode(data, validate=True))) as image:
                        image = image.convert('RGB')
                except (ValueError, binascii.Error, UnidentifiedImageError, OSError) as exc:
                    raise HTTPException(400, 'Invalid image payload') from exc
                images.append(image)
                parts.append({'type': 'image', 'image': image})
            else:
                raise HTTPException(400, 'Only text and user image_url parts are supported')
        resolved.append({'role': role, 'content': parts})
    if resolved[-1]['role'] != 'user':
        raise HTTPException(400, 'The last message must be a user message')
    return resolved, images


def build_app(config: ServiceConfig) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        service = Qwen3VLService(config)
        app.state.service = service
        try:
            await run_in_threadpool(service.load)
            yield
        finally:
            service.close()

    app = FastAPI(title='RTV temporary inference', lifespan=lifespan)

    @app.get('/healthz')
    def healthz():
        service = app.state.service
        return {'pid': os.getpid(), 'model': config.model,
                'loaded': service.model is not None and service.processor is not None}

    @app.post('/v1/chat/completions')
    async def completions(payload: ChatRequest):
        if payload.model != config.model:
            raise HTTPException(404, 'Requested model is not loaded by this evaluation')
        if payload.stream:
            raise HTTPException(400, 'The evaluation API does not use streaming')
        messages, images = resolve_messages(payload.messages)
        try:
            text = await run_in_threadpool(
                app.state.service.infer_messages,
                messages=messages, images=images,
                max_new_tokens=payload.max_tokens, temperature=payload.temperature,
            )
        finally:
            for image in images:
                image.close()
        return {
            'id': f'chatcmpl-{uuid.uuid4().hex}', 'object': 'chat.completion',
            'created': int(time.time()), 'model': config.model,
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text},
                         'finish_reason': 'stop'}],
        }

    return app


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    config = ServiceConfig(**json.loads(os.environ['RTV_LOCAL_MODEL_CONFIG']))
    uvicorn.run(build_app(config), host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
