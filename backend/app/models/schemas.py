"""数据模型定义"""
# -*- coding: utf-8 -*-
from typing import List, Optional, Union
from pydantic import BaseModel, Field, field_validator


# ============ 请求模型 ============

class TripRequest(BaseModel):
    """旅行规划请求"""
    nickname: str = Field(..., description="用户昵称,用于生成长期用户记忆", example="Alice")
    departure_city: str = Field(default="", description="出发城市", example="天津")
    city: str = Field(..., description="目的地城市", example="北京")
    start_date: str = Field(..., description="开始日期 YYYY-MM-DD", example="2025-06-01")
    end_date: str = Field(..., description="结束日期 YYYY-MM-DD", example="2025-06-03")
    travel_days: int = Field(..., description="旅行天数", ge=1, le=30, example=3)
    intercity_transportation: str = Field(default="智能推荐", description="大交通偏好: 智能推荐/飞机/火车/自驾", example="智能推荐")
    transportation: str = Field(..., description="交通方式", example="公共交通")
    accommodation: str = Field(..., description="住宿偏好", example="经济型酒店")
    preferences: List[str] = Field(default=[], description="旅行偏好标签", example=["历史文化", "美食"])
    free_text_input: Optional[str] = Field(default="", description="额外要求", example="希望多安排一些博物馆")
    
    class Config:
        json_schema_extra = {
            "example": {
                "nickname": "Alice",
                "departure_city": "天津",
                "city": "北京",
                "start_date": "2025-06-01",
                "end_date": "2025-06-03",
                "travel_days": 3,
                "intercity_transportation": "智能推荐",
                "transportation": "公共交通",
                "accommodation": "经济型酒店",
                "preferences": ["历史文化", "美食"],
                "free_text_input": "希望多安排一些博物馆"
            }
        }


class TripChatRequest(BaseModel):
    """旅行计划多轮对话请求"""
    task_id: str = Field(..., description="任务ID", example="task_20260423_ab12cd34")
    user_message: str = Field(..., description="用户本轮修改意见", example="把第二天下午换成博物馆")


class POISearchRequest(BaseModel):
    """POI搜索请求"""
    keywords: str = Field(..., description="搜索关键词", example="故宫")
    city: str = Field(..., description="城市", example="北京")
    citylimit: bool = Field(default=True, description="是否限制在城市范围内")


class RouteRequest(BaseModel):
    """路线规划请求"""
    origin_address: str = Field(..., description="起点地址", example="北京市朝阳区阜通东大街6号")
    destination_address: str = Field(..., description="终点地址", example="北京市海淀区上地十街10号")
    origin_city: Optional[str] = Field(default=None, description="起点城市")
    destination_city: Optional[str] = Field(default=None, description="终点城市")
    route_type: str = Field(default="walking", description="路线类型: walking/driving/transit")


# ============ 响应模型 ============

class Location(BaseModel):
    """地理位置"""
    longitude: float = Field(..., description="经度")
    latitude: float = Field(..., description="纬度")


class Attraction(BaseModel):
    """景点信息"""
    name: str = Field(..., description="景点名称")
    address: str = Field(..., description="地址")
    location: Location = Field(..., description="经纬度坐标")
    visit_duration: int = Field(..., description="建议游览时间(分钟)")
    description: str = Field(..., description="景点描述")
    category: Optional[str] = Field(default="景点", description="景点类别")
    rating: Optional[float] = Field(default=None, description="评分")
    photos: Optional[List[str]] = Field(default_factory=list, description="景点图片URL列表")
    poi_id: Optional[str] = Field(default="", description="POI ID")
    image_url: Optional[str] = Field(default=None, description="图片URL")
    ticket_price: int = Field(default=0, description="门票价格(元)")


class Meal(BaseModel):
    """餐饮信息"""
    type: str = Field(..., description="餐饮类型: breakfast/lunch/dinner/snack")
    name: str = Field(..., description="餐饮名称")
    address: Optional[str] = Field(default=None, description="地址")
    location: Optional[Location] = Field(default=None, description="经纬度坐标")
    description: Optional[str] = Field(default=None, description="描述")
    estimated_cost: int = Field(default=0, description="预估费用(元)")


class Hotel(BaseModel):
    """酒店信息"""
    name: str = Field(..., description="酒店名称")
    address: str = Field(default="", description="酒店地址")
    location: Optional[Location] = Field(default=None, description="酒店位置")
    price_range: str = Field(default="", description="价格范围")
    rating: str = Field(default="", description="评分")
    distance: str = Field(default="", description="距离景点距离")
    type: str = Field(default="", description="酒店类型")
    estimated_cost: int = Field(default=0, description="预估费用(元/晚)")
    price_source: str = Field(default="estimate", description="价格来源: tuniu_detail_price/tuniu_lowest_price/amap_cost/llm_estimate/default_estimate")


class TransportSegment(BaseModel):
    """单段交通预算信息"""
    from_name: str = Field(..., description="起点名称")
    to_name: str = Field(..., description="终点名称")
    mode: str = Field(..., description="交通方式")
    distance: float = Field(default=0, description="距离(米)")
    duration: int = Field(default=0, description="时间(秒)")
    estimated_cost: int = Field(default=0, description="预估费用(元)")
    cost_source: str = Field(default="rule_based", description="费用来源: route_fee/route_estimate/rule_based")
    description: str = Field(default="", description="路线说明")


class IntercityTransportOption(BaseModel):
    """城市间大交通候选."""

    direction: str = Field(default="", description="方向: outbound/return")
    mode: str = Field(default="", description="交通方式: 飞机/火车/自驾")
    provider: str = Field(default="", description="数据提供方")
    departure_city: str = Field(default="", description="出发城市")
    arrival_city: str = Field(default="", description="到达城市")
    date: str = Field(default="", description="出行日期 YYYY-MM-DD")
    departure_time: str = Field(default="", description="出发时间")
    arrival_time: str = Field(default="", description="到达时间")
    duration_minutes: int = Field(default=0, description="耗时(分钟)")
    estimated_cost: int = Field(default=0, description="预估费用(元)")
    code: str = Field(default="", description="航班号/车次")
    data_source: str = Field(default="", description="价格/路线数据来源")
    description: str = Field(default="", description="补充说明")


class IntercityTransportPlan(BaseModel):
    """城市间大交通计划."""

    status: str = Field(default="skipped", description="状态: skipped/ok/partial/unavailable")
    preference: str = Field(default="智能推荐", description="大交通偏好")
    outbound_candidates: List[IntercityTransportOption] = Field(default_factory=list)
    return_candidates: List[IntercityTransportOption] = Field(default_factory=list)
    selected_outbound: Optional[IntercityTransportOption] = None
    selected_return: Optional[IntercityTransportOption] = None
    schedule_constraints: dict = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


class DayBudget(BaseModel):
    """单日预算信息"""
    attractions: int = Field(default=0, description="当日景点门票费用")
    meals: int = Field(default=0, description="当日餐饮费用")
    hotel: int = Field(default=0, description="当日酒店费用")
    transportation: int = Field(default=0, description="当日交通费用")
    subtotal: int = Field(default=0, description="当日费用小计")


class DayPlan(BaseModel):
    """单日行程"""
    date: str = Field(..., description="日期 YYYY-MM-DD")
    day_index: int = Field(..., description="第几天(从0开始)")
    description: str = Field(..., description="当日行程描述")
    transportation: str = Field(..., description="交通方式")
    accommodation: str = Field(..., description="住宿")
    hotel: Optional[Hotel] = Field(default=None, description="推荐酒店")
    attractions: List[Attraction] = Field(default=[], description="景点列表")
    meals: List[Meal] = Field(default=[], description="餐饮列表")
    transport_segments: List[TransportSegment] = Field(default_factory=list, description="当日交通段")
    day_budget: DayBudget = Field(default_factory=DayBudget, description="当日预算")


class WeatherInfo(BaseModel):
    """天气信息"""
    date: str = Field(..., description="日期 YYYY-MM-DD")
    day_weather: str = Field(default="", description="白天天气")
    night_weather: str = Field(default="", description="夜间天气")
    day_temp: Union[int, str] = Field(default=0, description="白天温度")
    night_temp: Union[int, str] = Field(default=0, description="夜间温度")
    wind_direction: str = Field(default="", description="风向")
    wind_power: str = Field(default="", description="风力")

    @field_validator('day_temp', 'night_temp', mode='before')
    @classmethod
    def parse_temperature(cls, v):
        """解析温度,移除°C等单位"""
        if isinstance(v, str):
            # 移除°C, ℃等单位符号
            v = v.replace('°C', '').replace('℃', '').replace('°', '').strip()
            try:
                return int(v)
            except ValueError:
                return 0
        return v


class Budget(BaseModel):
    """预算信息"""
    total_attractions: int = Field(default=0, description="景点门票总费用")
    total_hotels: int = Field(default=0, description="酒店总费用")
    total_meals: int = Field(default=0, description="餐饮总费用")
    total_transportation: int = Field(default=0, description="交通总费用")
    total_intercity_transportation: int = Field(default=0, description="大交通总费用")
    total: int = Field(default=0, description="总费用")


class TripPlan(BaseModel):
    """旅行计划"""
    departure_city: str = Field(default="", description="出发城市")
    city: str = Field(..., description="目的地城市")
    start_date: str = Field(..., description="开始日期")
    end_date: str = Field(..., description="结束日期")
    days: List[DayPlan] = Field(..., description="每日行程")
    weather_info: List[WeatherInfo] = Field(default=[], description="天气信息")
    overall_suggestions: str = Field(..., description="总体建议")
    budget: Optional[Budget] = Field(default=None, description="预算信息")
    intercity_transport: Optional[IntercityTransportPlan] = Field(default=None, description="城市间大交通计划")


class TripPlanResponse(BaseModel):
    """旅行计划响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(default="", description="消息")
    task_id: Optional[str] = Field(default=None, description="任务ID")
    user_id: Optional[str] = Field(default=None, description="用户ID")
    update_mode: str = Field(default="initial", description="更新模式: initial/patch/replan/restore")
    assistant_message: str = Field(default="", description="智能体给用户的简短说明")
    data: Optional[TripPlan] = Field(default=None, description="旅行计划数据")


class POIInfo(BaseModel):
    """POI信息"""
    id: str = Field(..., description="POI ID")
    name: str = Field(..., description="名称")
    type: str = Field(..., description="类型")
    address: str = Field(..., description="地址")
    location: Location = Field(..., description="经纬度坐标")
    tel: Optional[str] = Field(default=None, description="电话")


class POISearchResponse(BaseModel):
    """POI搜索响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(default="", description="消息")
    data: List[POIInfo] = Field(default=[], description="POI列表")


class RouteInfo(BaseModel):
    """路线信息"""
    distance: float = Field(..., description="距离(米)")
    duration: int = Field(..., description="时间(秒)")
    route_type: str = Field(..., description="路线类型")
    description: str = Field(..., description="路线描述")


class RouteResponse(BaseModel):
    """路线规划响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(default="", description="消息")
    data: Optional[RouteInfo] = Field(default=None, description="路线信息")


class WeatherResponse(BaseModel):
    """天气查询响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(default="", description="消息")
    data: List[WeatherInfo] = Field(default=[], description="天气信息")


# ============ 错误响应 ============

class ErrorResponse(BaseModel):
    """错误响应"""
    success: bool = Field(default=False, description="是否成功")
    message: str = Field(..., description="错误消息")
    error_code: Optional[str] = Field(default=None, description="错误代码")

