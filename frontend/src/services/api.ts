import axios from 'axios'
import type { TripChatRequest, TripFormData, TripPlanResponse } from '@/types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5分钟超时
  headers: {
    'Content-Type': 'application/json'
  }
})

export interface TripStreamEvent {
  event: string
  data: any
}

export type TripStreamHandler = (event: TripStreamEvent) => void

function parseSseBlock(block: string): TripStreamEvent | null {
  const lines = block.split(/\r?\n/)
  let event = 'message'
  const dataLines: string[] = []

  for (const line of lines) {
    if (line.startsWith('event:')) {
      event = line.slice(6).trim()
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trimStart())
    }
  }

  if (!dataLines.length) return null
  const dataText = dataLines.join('\n')
  return {
    event,
    data: dataText ? JSON.parse(dataText) : {}
  }
}

async function postSseTrip<TPayload>(
  url: string,
  payload: TPayload,
  onEvent?: TripStreamHandler
): Promise<TripPlanResponse> {
  const response = await fetch(`${API_BASE_URL}${url}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream'
    },
    body: JSON.stringify(payload)
  })

  if (!response.ok) {
    let detail = response.statusText
    try {
      const data = await response.json()
      detail = data?.detail || detail
    } catch {
      // Keep the HTTP status text when the response is not JSON.
    }
    throw new Error(detail || '流式请求失败')
  }
  if (!response.body) {
    throw new Error('当前浏览器不支持流式响应')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let finalResult: TripPlanResponse | null = null

  const handleBlock = (block: string) => {
    const parsed = parseSseBlock(block.trim())
    if (!parsed) return
    onEvent?.(parsed)
    if (parsed.event === 'result') {
      finalResult = parsed.data as TripPlanResponse
    }
    if (parsed.event === 'error') {
      throw new Error(parsed.data?.message || '流式请求失败')
    }
  }

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const parts = buffer.split(/\r?\n\r?\n/)
    buffer = parts.pop() || ''
    for (const part of parts) {
      handleBlock(part)
    }
  }

  buffer += decoder.decode()
  if (buffer.trim()) {
    handleBlock(buffer)
  }

  if (!finalResult) {
    throw new Error('流式响应未返回最终计划')
  }
  return finalResult
}

// 请求拦截器
apiClient.interceptors.request.use(
  (config) => {
    console.log('发送请求:', config.method?.toUpperCase(), config.url)
    return config
  },
  (error) => {
    console.error('请求错误:', error)
    return Promise.reject(error)
  }
)

// 响应拦截器
apiClient.interceptors.response.use(
  (response) => {
    console.log('收到响应:', response.status, response.config.url)
    return response
  },
  (error) => {
    console.error('响应错误:', error.response?.status, error.message)
    return Promise.reject(error)
  }
)

/**
 * 生成旅行计划
 */
export async function generateTripPlan(formData: TripFormData): Promise<TripPlanResponse> {
  try {
    const response = await apiClient.post<TripPlanResponse>('/api/trip/plan', formData)
    return response.data
  } catch (error: any) {
    console.error('生成旅行计划失败:', error)
    throw new Error(error.response?.data?.detail || error.message || '生成旅行计划失败')
  }
}

export async function generateTripPlanStream(
  formData: TripFormData,
  onEvent?: TripStreamHandler
): Promise<TripPlanResponse> {
  try {
    return await postSseTrip('/api/trip/plan/stream', formData, onEvent)
  } catch (error: any) {
    console.error('流式生成旅行计划失败:', error)
    throw new Error(error.message || '流式生成旅行计划失败')
  }
}

/**
 * 多轮对话修改旅行计划
 */
export async function chatWithTrip(payload: TripChatRequest): Promise<TripPlanResponse> {
  try {
    const response = await apiClient.post<TripPlanResponse>('/api/trip/chat', payload)
    return response.data
  } catch (error: any) {
    console.error('聊天修改计划失败:', error)
    throw new Error(error.response?.data?.detail || error.message || '聊天修改计划失败')
  }
}

export async function chatWithTripStream(
  payload: TripChatRequest,
  onEvent?: TripStreamHandler
): Promise<TripPlanResponse> {
  try {
    return await postSseTrip('/api/trip/chat/stream', payload, onEvent)
  } catch (error: any) {
    console.error('流式聊天修改计划失败:', error)
    throw new Error(error.message || '流式聊天修改计划失败')
  }
}

/**
 * 根据task_id恢复任务快照
 */
export async function getTripTask(taskId: string): Promise<TripPlanResponse> {
  try {
    const response = await apiClient.get<TripPlanResponse>(`/api/trip/task/${encodeURIComponent(taskId)}`)
    return response.data
  } catch (error: any) {
    console.error('恢复任务失败:', error)
    throw new Error(error.response?.data?.detail || error.message || '恢复任务失败')
  }
}

/**
 * 健康检查
 */
export async function healthCheck(): Promise<any> {
  try {
    const response = await apiClient.get('/health')
    return response.data
  } catch (error: any) {
    console.error('健康检查失败:', error)
    throw new Error(error.message || '健康检查失败')
  }
}

export default apiClient

