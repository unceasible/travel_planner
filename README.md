# HelloAgents 智能旅行助手

自研多智能体协作系统，采用异步并行流水线架构，集成高德地图、途牛票务与 Unsplash 等实时数据源，提供端到端的个性化旅行规划与多轮对话修改体验。LLM 接入与 MCP 工具管理借力 HelloAgents 的 `HelloAgentsLLM` 与 `MCPTool` 两个工具类。

## 核心架构

### 多智能体协作

系统由 **9 个智能体角色** 组成，分为三层：规划与检索层、质量保障层、对话感知层。所有智能体均为自研 Python 类，HelloAgents 仅提供 LLM 连接（`HelloAgentsLLM`）与 MCP 工具管理（`MCPTool`）两个基础设施类。

#### 智能体清单

**规划与检索层（5 个 Agent）**

| 智能体 | 模型 | 文件 | 职责 |
|--------|------|------|------|
| `MultiAgentTripPlanner` | 主模型 | `agents/trip_planner_agent.py` | 总调度，内含 4 个领域子 Agent 并行检索，最终通过 LLM 结构化输出生成完整 `TripPlan` |
| ↳ 景点子 Agent | — | 同上（prompt 模板） | 调用高德 POI 搜索 API，按偏好关键词 + 动态扩展多轮检索景点候选 |
| ↳ 酒店子 Agent | — | 同上（prompt 模板） | 途牛 MCP 实时价格查询 → 高德 POI 补充，地理聚类优选 |
| ↳ 天气子 Agent | — | 同上（prompt 模板） | 调用高德天气 API，返回行程日期范围内的逐日天气预报 |
| ↳ 餐饮子 Agent | — | 同上（prompt 模板） | 按菜系偏好搜索餐厅，关键词无结果时自动 fallback 泛化检索 |
| `IntercityTransportAgent` | — | `services/intercity_transport_agent.py` | 并行检索途牛机票/火车票 MCP + 高德自驾路线，自动优选并产出到达/离开时间约束 |

**质量保障层（1 个 Agent）**

| 智能体 | 模型 | 文件 | 职责 |
|--------|------|------|------|
| `ReflectionAgent` | 主模型 | `agents/reflection_agent.py` | 对已生成计划按 8 个维度结构化审阅（需求匹配 / 偏好满足 / 行程密度 / 地理路线合理性 / 餐饮酒店景点混用 / 天气影响 / 预算合理性 / 修改请求执行度），输出 0-10 分 + 具体问题列表 + 重修建议 |

**对话感知层（3 个 Agent）**

| 智能体 | 模型 | 文件 | 职责 |
|--------|------|------|------|
| `IntentClassifier` | 廉价模型 + Embedding | `services/intent_classifier.py` | 三级级联分类用户意图：正则规则（30+ 条）→ SiliconFlow Embedding 余弦相似度 → LLM JSON 分类，输出 `primary_intent` + `domains` + `action` |
| `UserProfileAgent` | 廉价模型 | `agents/user_profile_agent.py` | 后台异步运行，从用户消息中提取长期偏好 / 忌口 / 预算敏感度，写入 MemoryStore 供后续任务复用 |
| `ConversationContextCompressor` | 廉价模型 | `services/conversation_context.py` | 对话过长时后台分段并行总结，合并后存入 MemoryStore 的 `conversation_context` 字段 |

#### 调度机制：多线程池隔离

`TripTaskExecutor`（`services/task_executor.py`）是整个系统的中央编排器，内部维护 **5 个独立 `ThreadPoolExecutor`**，按任务类型隔离：

| 线程池 | 大小 | 用途 |
|--------|------|------|
| `retrieval_executor` | 5 | 初始规划时并行提交 5 路检索（景点 / 酒店 / 餐饮 / 天气 / 大交通）；chat 修改时按需并行提交 |
| `transport_executor` | 8 | 逐日逐段并行计算交通段（高德路线 API），含缓存去重 |
| `route_executor` | 可配（默认 1） | 高德路线规划 API 调用，受速率限制保护 |
| `profile_executor` | 1 | 后台异步执行 `UserProfileAgent.update_profile()`，不阻塞主流程 |
| `context_executor` | 1 | 后台异步执行 `ConversationContextCompressor.refresh_heavy_summary()` |

调度原则：
- **前台阻塞**：检索与 LLM 生成阶段同步等待结果，确保计划完整性
- **后台 fire-and-forget**：用户画像更新与上下文压缩提交后立即返回，失败仅记日志不阻断主流程
- **并行度受控**：每个池有独立线程上限，高德路线 API 池额外有速率限制（`amap_route_min_interval_seconds`）与指数退避重试

#### 通信模式

智能体之间**不直接通信**，全部通过中央编排器 `TripTaskExecutor` 以 **共享状态 + 结构化数据传递** 方式协同：

```
                    ┌──────────────────────────────┐
                    │        MemoryStore            │
                    │  (Markdown 文件 + 内存缓存)    │
                    │  task/{id}.md  users/{id}.md  │
                    └──────────┬───────────────────┘
                               │ read / write
                    ┌──────────┴───────────────────┐
                    │      TripTaskExecutor         │
                    │  (中央编排器)                  │
                    └──────────────────────────────┘
                     │         │         │
                     ▼         ▼         ▼
              TripPlanner  Reflection  UserProfile
               Agent        Agent       Agent
```

1. **Pydantic 结构化契约**：所有 Agent 的输入输出均为强类型 Pydantic 模型（`TripPlan`, `TripRequest`, `ReflectionReview`, `IntentResult` 等），编排器负责在 Agent 间做格式转换，不存在松散字符串拼接
2. **MemoryStore 共享状态**：任务快照（`current_plan`, `conversation_log`, `reflection_log`, `form_snapshot`）和用户画像（`preferences`, `constraints`）持久化为 Markdown 文件，各 Agent 通过编排器间接读写，支持跨请求恢复
3. **SSE 进度推送**：编排器在每个阶段切换时通过回调 `ProgressCallback` 推送 `stage` + `percent` + `message`，经 FastAPI `StreamingResponse` 以 `text/event-stream` 实时送达前端
4. **质量门禁反馈闭环**：`ReflectionAgent` 输出的 `improvement_instructions` 作为 `reflection_feedback` 参数回传给 `TripPlannerAgent` 的重试调用，形成 Plan → Review → Replan 的修正环

### 异步并行流水线

每个旅行规划请求经历一条 **多阶段异步流水线**，各阶段内部以线程池并行执行：

```
创建任务 ─→ 并行检索(景点‖酒店‖餐饮‖天气‖大交通) ─→ LLM 结构化生成
   │                                                      │
   └─ 后台: 用户画像更新(异步)                              │
                                                          ↓
   保存 ←── 质量门禁 ←── 预算汇总 ←── 并行交通段计算(逐日多段并行)
```

- **并行检索**：5 路并发（景点 / 酒店 / 餐饮 / 天气 / 城际交通），`ThreadPoolExecutor` 管理
- **并行交通**：逐日逐段并发调用高德路线规划 API，含缓存去重
- **异步画像更新**：`UserProfileAgent` 提交到独立线程池，不阻塞规划主流程
- **异步上下文压缩**：对话过长时后台并行总结历史，多分段 LLM 调用后合并

### 质量保障机制

- **质量门禁**：`ReflectionAgent` 评分 < 7 时自动携带反馈重规划一次
- **格式修正**：天数补齐、缺失餐食补充、酒店跨天继承、天气信息补全
- **交通偏好守卫**：强制过滤不匹配的自驾建议
- **不安全酒店移除**：与当日景点地理距离超阈值时自动剔除

## 技术栈

### 后端
- **智能体框架**：自研多智能体编排 + HelloAgents 工具类（`HelloAgentsLLM` / `MCPTool`）
- **API 框架**：FastAPI + SSE 流式推送
- **数据源**：高德地图 MCP、途牛酒店/机票/火车票 MCP、Unsplash 图片
- **LLM**：OpenAI 兼容接口，支持主模型 + 廉价模型双配置
- **并发**：`ThreadPoolExecutor` 多池隔离（检索 / 交通 / 画像 / 压缩，各池独立）
- **存储**：Markdown 文件持久化任务与用户记忆，支持异步写入

### 前端
- **框架**：Vue 3 + TypeScript + Vite
- **UI 组件库**：Ant Design Vue
- **地图服务**：高德地图 JavaScript API
- **导出**：html2canvas + jsPDF 生成 PDF 行程单

## 项目结构

```
travel_planner/
├── backend/
│   ├── app/
│   │   ├── agents/                    # 多智能体定义
│   │   │   ├── trip_planner_agent.py  #   主规划智能体（检索 + 结构化生成）
│   │   │   ├── reflection_agent.py    #   质量审阅智能体
│   │   │   └── user_profile_agent.py  #   用户画像智能体（异步）
│   │   ├── services/                  # 服务层
│   │   │   ├── task_executor.py       #   任务编排器（流水线核心）
│   │   │   ├── intercity_transport_agent.py  # 城际大交通智能体
│   │   │   ├── intent_classifier.py   #   意图分类器（规则 + Embedding + LLM）
│   │   │   ├── conversation_context.py #  对话上下文压缩
│   │   │   ├── memory_store.py        #   任务与用户记忆持久化
│   │   │   ├── amap_service.py        #   高德地图 MCP 封装
│   │   │   ├── amap_tool_pool.py      #   高德 MCP 工具池
│   │   │   ├── tuniu_hotel_service.py #   途牛酒店 MCP
│   │   │   └── llm_service.py         #   LLM 客户端工厂
│   │   ├── api/
│   │   │   ├── main.py                #   FastAPI 应用入口
│   │   │   └── routes/
│   │   │       ├── trip.py            #   旅行规划 API（含 SSE 流式端点）
│   │   │       ├── poi.py             #   POI 详情与图片
│   │   │       └── map.py             #   地图服务（POI/天气/路线）
│   │   ├── models/
│   │   │   └── schemas.py             #   Pydantic 数据模型
│   │   └── config.py                  #   配置管理（pydantic-settings）
│   ├── tests/                         # 单元测试
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── components/                # Vue 组件
│   │   ├── services/                  # API 服务层
│   │   └── views/                     # 页面视图
│   ├── package.json
│   └── vite.config.ts
└── README.md
```

## 快速开始

### 前提条件

- Python 3.10+
- Node.js 16+
- 高德地图 API Key（Web 服务 API + Web 端 JS API）
- LLM API Key（OpenAI 兼容接口）
- 途牛 API Key（可选，用于酒店实时价格与机票火车票检索）
- SiliconFlow API Key（可选，用于意图分类 Embedding 加速）
- Unsplash API Key（可选，用于景点图片）

### 后端

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # 编辑 .env 填入各 API Key
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 前端

```bash
cd frontend
npm install
cp .env.example .env       # 填入高德地图 Web API Key 与 JS API Key
npm run dev                # 访问 http://localhost:5173
```

## API 概览

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/trip/plan` | POST | 生成旅行计划（同步） |
| `/api/trip/plan/stream` | POST | 生成旅行计划（SSE 流式，支持进度推送） |
| `/api/trip/chat` | POST | 多轮对话修改计划（同步） |
| `/api/trip/chat/stream` | POST | 多轮对话修改计划（SSE 流式） |
| `/api/trip/task/{task_id}` | GET | 恢复任务快照 |
| `/api/poi/detail/{poi_id}` | GET | POI 详情与图片 |
| `/api/poi/search` | GET | POI 搜索 |
| `/api/poi/photo` | GET | Unsplash 景点图片 |
| `/api/map/poi` | GET | 高德 POI 搜索 |
| `/api/map/weather` | GET | 天气查询 |
| `/api/map/route` | POST | 路线规划 |

启动后访问 `http://localhost:8000/docs` 查看完整 Swagger 文档。

## SSE 进度事件

流式端点按阶段推送 `progress` 事件，前端可实时展示当前流水线状态：

```
task_created → profile_update_started → retrieval_started → retrieval_completed
→ llm_started → llm_completed → transport_started → budget_started
→ reflection_started → persisted → result
```

## 配置

核心环境变量（详见 `.env.example`）：

| 变量 | 说明 |
|------|------|
| `AMAP_API_KEY` | 高德地图 Web 服务 API Key（必填） |
| `LLM_API_KEY` / `OPENAI_API_KEY` | 主 LLM API Key |
| `LLM_BASE_URL` | 主 LLM 接口地址 |
| `LLM_MODEL_ID` | 主模型名称（用于规划与审阅） |
| `CHEAP_MODEL` | 廉价模型标识（用于意图分类、画像更新、上下文压缩） |
| `TUNIU_API_KEY` | 途牛 API Key（用于酒店实时价格与票务） |
| `SILICONFLOW_API_KEY` | SiliconFlow API Key（用于 Embedding 意图匹配） |
| `UNSPLASH_ACCESS_KEY` | Unsplash API Key（用于景点图片） |

## 开源协议

CC BY-NC-SA 4.0

## 致谢

- [HelloAgents](https://github.com/datawhalechina/Hello-Agents) — 智能体教程
- [HelloAgents 框架](https://github.com/jjyaoao/HelloAgents) — 智能体框架
- [高德地图开放平台](https://lbs.amap.com/) — 地图服务
- [途牛开放平台](https://open.tuniu.cn/) — 酒店与票务 MCP
- [Unsplash](https://unsplash.com/developers) — 图片服务
