"""旅行规划API路由."""

import json
import queue
import threading
import time
import traceback
from typing import Any, Callable

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from ...models.schemas import TripChatRequest, TripPlanResponse, TripRequest
from ...services.agent_output_logger import log_event, timed_event
from ...services.task_executor import get_trip_task_executor

router = APIRouter(prefix="/trip", tags=["旅行规划"])

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _encode_sse(event: str, data: Any) -> str:
    payload = json.dumps(jsonable_encoder(data), ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _stream_trip_response(
    kind: str,
    task_id_hint: str,
    worker: Callable[[Callable[[str, dict[str, Any]], None]], TripPlanResponse],
) -> StreamingResponse:
    event_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
    started_at = time.perf_counter()

    def log_sse_event(event: str, payload: Any) -> None:
        if event == "heartbeat":
            return
        data = payload if isinstance(payload, dict) else {}
        log_event(
            f"sse.{kind}.event",
            {
                "event": event,
                "stage": data.get("stage", event),
                "task_id": data.get("task_id") or task_id_hint,
                "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
            },
        )

    def push_event(event: str, payload: Any) -> None:
        log_sse_event(event, payload)
        event_queue.put((event, payload))

    def progress(stage: str, payload: dict[str, Any]) -> None:
        data = dict(payload or {})
        data.setdefault("stage", stage)
        push_event("progress", data)

    def run_worker() -> None:
        try:
            result = worker(progress)
            push_event("result", result.model_dump())
        except Exception as exc:
            log_event(
                "sse.stream.error",
                {
                    "kind": kind,
                    "task_id": task_id_hint,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(limit=5),
                    "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
                },
            )
            push_event("error", {"message": str(exc), "stage": "error", "task_id": task_id_hint})
        finally:
            event_queue.put(("done", {}))

    def event_generator():
        thread = threading.Thread(target=run_worker, daemon=True)
        thread.start()
        while True:
            try:
                event, payload = event_queue.get(timeout=12)
            except queue.Empty:
                yield _encode_sse(
                    "heartbeat",
                    {
                        "stage": "heartbeat",
                        "task_id": task_id_hint,
                        "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
                    },
                )
                continue
            if event == "done":
                yield _encode_sse("done", {})
                break
            yield _encode_sse(event, payload)

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=SSE_HEADERS)


@router.post(
    "/plan",
    response_model=TripPlanResponse,
    summary="生成旅行计划",
    description="根据用户输入的旅行需求,生成详细的旅行计划",
)
async def plan_trip(request: TripRequest):
    """生成初次旅行计划."""
    try:
        executor = get_trip_task_executor()
        with timed_event("api.trip.plan.total", {"city": request.city, "travel_days": request.travel_days}):
            return executor.plan_initial(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成旅行计划失败: {e}")


@router.post(
    "/plan/stream",
    summary="流式生成旅行计划",
    description="通过 text/event-stream 返回旅行计划生成阶段进度,最终通过 result 事件返回完整计划",
)
async def plan_trip_stream(request: TripRequest):
    """流式生成初次旅行计划."""
    executor = get_trip_task_executor()

    def worker(progress: Callable[[str, dict[str, Any]], None]) -> TripPlanResponse:
        with timed_event("api.trip.plan.stream.total", {"city": request.city, "travel_days": request.travel_days}):
            return executor.plan_initial(request, progress=progress)

    return _stream_trip_response("plan", "", worker)


@router.post(
    "/chat",
    response_model=TripPlanResponse,
    summary="多轮对话修改计划",
    description="根据用户文字建议,对已有任务计划进行patch或replan",
)
async def chat_trip(request: TripChatRequest):
    """通过聊天修改现有任务计划."""
    try:
        executor = get_trip_task_executor()
        with timed_event("api.trip.chat.total", {"task_id": request.task_id}):
            return executor.chat(request)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新旅行计划失败: {e}")


@router.post(
    "/chat/stream",
    summary="流式修改旅行计划",
    description="通过 text/event-stream 返回聊天修改阶段进度,最终通过 result 事件返回完整计划",
)
async def chat_trip_stream(request: TripChatRequest):
    """流式修改现有任务计划."""
    executor = get_trip_task_executor()

    def worker(progress: Callable[[str, dict[str, Any]], None]) -> TripPlanResponse:
        with timed_event("api.trip.chat.stream.total", {"task_id": request.task_id}):
            return executor.chat(request, progress=progress)

    return _stream_trip_response("chat", request.task_id, worker)


@router.get(
    "/task/{task_id}",
    response_model=TripPlanResponse,
    summary="恢复任务计划",
    description="根据task_id加载最新旅行计划快照",
)
async def restore_task(task_id: str):
    """恢复指定任务的当前计划快照."""
    try:
        executor = get_trip_task_executor()
        return executor.restore(task_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"恢复任务失败: {e}")


@router.get(
    "/health",
    summary="健康检查",
    description="检查旅行规划服务是否正常",
)
async def health_check():
    """健康检查."""
    try:
        executor = get_trip_task_executor()
        return {
            "status": "healthy",
            "service": "trip-planner",
            "planner_ready": executor.planner is not None,
            "memory_store_ready": executor.memory_store is not None,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"服务不可用: {e}")
