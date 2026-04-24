"""旅行规划API路由."""

import traceback

from fastapi import APIRouter, HTTPException

from ...models.schemas import TripChatRequest, TripPlanResponse, TripRequest
from ...services.task_executor import get_trip_task_executor

router = APIRouter(prefix="/trip", tags=["旅行规划"])


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
        return executor.plan_initial(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成旅行计划失败: {e}")


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
        return executor.chat(request)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新旅行计划失败: {e}")


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
