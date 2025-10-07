from argparse import Namespace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Optional

import uvloop
import vllm.envs as envs
from fastapi import Request
from fastapi import UploadFile, File, Form, HTTPException
import os, shutil, json, tempfile, asyncio
from vllm.config import LogprobsMode
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.cli.serve import run_headless, run_multi_api_server
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    build_app,
    build_async_engine_client_from_engine_args,
    init_app_state,
    load_log_config,
    setup_server,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser

from prime_rl.inference.config import InferenceConfig

logger = init_logger("vllm.entrypoints.openai.api_server")
_UPDATE_LOCK = asyncio.Lock()
# _STAGING_DIR = os.environ.get("PRIMERL_STAGING_DIR", "/tmp/prime-rl/staging")
_STAGING_DIR = os.environ.get("PRIMERL_STAGING_DIR", "/scratch/users/nus/e1604484/project/prime-rl/staging")
os.makedirs(_STAGING_DIR, exist_ok=True)


# Copied from vllm/entrypoints/openai/api_server.py
# Only difference is that we extend the engine args with our custom worker extension
@asynccontextmanager
async def custom_build_async_engine_client(
    args: Namespace,
    client_config: Optional[dict[str, Any]] = None,
) -> AsyncIterator[EngineClient]:
    # Context manager to handle engine_client lifecycle
    # Ensures everything is shutdown and cleaned up on error/exit
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_args.worker_extension_cls = "prime_rl.inference.vllm.worker.CheckpointWorker"
    engine_args.logprobs_mode = LogprobsMode.PROCESSED_LOGPROBS

    async with build_async_engine_client_from_engine_args(
        engine_args, disable_frontend_multiprocessing=args.disable_frontend_multiprocessing, client_config=client_config
    ) as engine:
        yield engine


# Copied from vllm/entrypoints/openai/api_server.py
# Only difference is that we inject custom routes and build async engine client differently
async def custom_run_server_worker(listen_address, sock, args, client_config=None, **uvicorn_kwargs) -> None:
    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    server_index = client_config.get("client_index", 0) if client_config else 0

    # Load logging config for uvicorn if specified
    log_config = load_log_config(args.log_config_file)
    if log_config is not None:
        uvicorn_kwargs["log_config"] = log_config

    async with custom_build_async_engine_client(args, client_config) as engine_client:
        app = build_app(args)

        ### CUSTOM ENDPOINTS ###
        # @app.post("/update_weights")
        # async def _update_weights(request: Request):
        #     data = await request.json()
        #     model_path = data.get("model_path")
        #     await engine_client.collective_rpc("update_weights", args=(model_path,))
        #     return {"status": "ok"}

        @app.post("/update_weights")
        async def _update_weights(
            request: Request,
            # multipart 路径（MVP：单文件 .pt）
            metadata: str | None = Form(default=None),
            file: UploadFile | None = File(default=None),
        ):
            """
            两种用法：
            1) 兼容旧 JSON：{"model_path": "/abs/path/to/pytorch_model.bin"}
            2) 新增 multipart：
            - metadata: JSON 字符串，如 {"step": 12, "format": "pt"}
            - file:     上传的二进制文件（单文件 .pt）
            """
            async with _UPDATE_LOCK:
                # 分支 1：multipart 上传
                if file is not None:
                    try:
                        meta = json.loads(metadata) if metadata else {}
                    except Exception:
                        raise HTTPException(status_code=400, detail="invalid metadata json")

                    step = int(meta.get("step", -1))
                    fmt = (meta.get("format") or "pt").lower()
                    if fmt != "pt":
                        # MVP 只支持 pt；后续再扩 safetensors
                        raise HTTPException(status_code=415, detail="MVP only supports format=pt")

                    # 以 step 命名临时文件；若无 step 就用 NamedTemporaryFile
                    if step >= 0:
                        tmp_path = os.path.join(_STAGING_DIR, f"step_{step}.bin")
                        f = open(tmp_path, "wb")
                        should_close = True
                    else:
                        tf = tempfile.NamedTemporaryFile(prefix="weights_", suffix=".bin", dir=_STAGING_DIR, delete=False)
                        tmp_path = tf.name
                        f = tf
                        should_close = False

                    # 流式落盘，避免把 1GB 全读进内存
                    try:
                        with f:
                            shutil.copyfileobj(file.file, f, length=8 << 20)  # 8 MiB block
                    except Exception as e:
                        # 落盘失败清理
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass
                        raise HTTPException(status_code=500, detail=f"write temp file failed: {e}")

                    # 调用引擎加载
                    try:
                        await engine_client.collective_rpc("update_weights", args=(tmp_path,))
                        return {"status": "ok", "serving_step": step if step >= 0 else None, "message": "activated from upload"}
                    except Exception as e:
                        # 加载失败，保持旧模型继续服务
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass
                        raise HTTPException(status_code=500, detail=f"update_weights failed: {e}")

                # 分支 2：兼容原 JSON（传本地路径）
                try:
                    data = await request.json()
                except Exception:
                    raise HTTPException(status_code=400, detail="no multipart file and invalid/empty JSON body")

                model_path = data.get("model_path")
                if not model_path:
                    raise HTTPException(status_code=400, detail="model_path is required when not uploading file")

                try:
                    await engine_client.collective_rpc("update_weights", args=(model_path,))
                    return {"status": "ok", "message": "activated from path"}
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"update_weights failed: {e}")

        @app.post("/reload_weights")
        async def _reload_weights(request: Request):
            await engine_client.collective_rpc("reload_weights")
            return {"status": "ok"}

        vllm_config = await engine_client.get_vllm_config()
        await init_app_state(engine_client, vllm_config, app.state, args)

        logger.info("Starting vLLM API server %d on %s", server_index, listen_address)
        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            # NOTE: When the 'disable_uvicorn_access_log' value is True,
            # no access log will be output.
            access_log=not args.disable_uvicorn_access_log,
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    try:
        await shutdown_task
    finally:
        sock.close()


# Copied from vllm/entrypoints/openai/api_server.py
# Only difference is that we call `custom_run_server_worker` instead of `run_server_worker`
async def custom_run_server(args: Namespace, **uvicorn_kwargs) -> None:
    listen_address, sock = setup_server(args)
    await custom_run_server_worker(listen_address, sock, args, **uvicorn_kwargs)


# Adapted from vllm/entrypoints/cli/serve.py
# Only difference is that we call `custom_run_server` instead of `run_server` and we do config translation (i.e. pass populated namespace to `parse_args`)
def server(config: InferenceConfig, vllm_args: list[str]):
    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args(args=vllm_args, namespace=config.to_vllm())
    validate_parsed_serve_args(args)

    # Raise error if logprobs_mode is not set to processed_logprobs
    if args.logprobs_mode != "processed_logprobs":
        raise ValueError("logprobs_mode must be 'processed_logprobs' to be compatible with the orchestrator.")

    if args.headless or args.api_server_count < 1:
        run_headless(args)
    else:
        if args.api_server_count > 1:
            run_multi_api_server(args)
        else:
            # Single API server (this process).
            uvloop.run(custom_run_server(args))
