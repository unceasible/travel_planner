"""高德地图MCP服务封装"""
# -*- coding: utf-8 -*-
import json
import re
from typing import List, Dict, Any, Optional
import httpx
from hello_agents.tools import MCPTool
from ..config import get_settings
from ..models.schemas import Location, POIInfo, WeatherInfo
from .agent_output_logger import log_event

# 全局MCP工具实例
_amap_mcp_tool = None


def get_amap_mcp_tool() -> MCPTool:
    """
    获取高德地图MCP工具实例(单例模式)
    
    Returns:
        MCPTool实例
    """
    global _amap_mcp_tool
    
    if _amap_mcp_tool is None:
        settings = get_settings()
        
        if not settings.amap_api_key:
            raise ValueError("高德地图API Key未配置,请在.env文件中设置AMAP_API_KEY")
        
        # 创建MCP工具
        _amap_mcp_tool = MCPTool(
            name="amap",
            description="高德地图服务,支持POI搜索、路线规划、天气查询等功能",
            server_command=["uvx", "amap-mcp-server"],
            env={"AMAP_MAPS_API_KEY": settings.amap_api_key},
            auto_expand=True  # 自动展开为独立工具
        )
        
        print(f"✅ 高德地图MCP工具初始化成功")
        print(f"   工具数量: {len(_amap_mcp_tool._available_tools)}")
        
        # 打印可用工具列表
        if _amap_mcp_tool._available_tools:
            print("   可用工具:")
            for tool in _amap_mcp_tool._available_tools[:5]:  # 只打印前5个
                print(f"     - {tool.get('name', 'unknown')}")
            if len(_amap_mcp_tool._available_tools) > 5:
                print(f"     ... 还有 {len(_amap_mcp_tool._available_tools) - 5} 个工具")
    
    return _amap_mcp_tool


class AmapService:
    """高德地图服务封装类"""
    
    def __init__(self):
        """初始化服务"""
        self.settings = get_settings()
        self.http_client = httpx.Client(timeout=12.0)
        self.mcp_tool = get_amap_mcp_tool()
    
    def search_poi(self, keywords: str, city: str, citylimit: bool = True) -> List[POIInfo]:
        """
        搜索POI
        
        Args:
            keywords: 搜索关键词
            city: 城市
            citylimit: 是否限制在城市范围内
            
        Returns:
            POI信息列表
        """
        try:
            # 调用MCP工具
            result = self.mcp_tool.run({
                "action": "call_tool",
                "tool_name": "maps_text_search",
                "arguments": {
                    "keywords": keywords,
                    "city": city,
                    "citylimit": str(citylimit).lower()
                }
            })
            
            # 解析结果
            # 注意: MCP工具返回的是字符串,需要解析
            # 这里简化处理,实际应该解析JSON
            print(f"POI搜索结果: {result[:200]}...")  # 打印前200字符
            
            # TODO: 解析实际的POI数据
            return []
            
        except Exception as e:
            print(f"❌ POI搜索失败: {str(e)}")
            return []
    
    def get_weather(self, city: str) -> List[WeatherInfo]:
        """
        查询天气
        
        Args:
            city: 城市名称
            
        Returns:
            天气信息列表
        """
        try:
            # 调用MCP工具
            result = self.mcp_tool.run({
                "action": "call_tool",
                "tool_name": "maps_weather",
                "arguments": {
                    "city": city
                }
            })
            
            print(f"天气查询结果: {result[:200]}...")
            
            # TODO: 解析实际的天气数据
            return []
            
        except Exception as e:
            print(f"❌ 天气查询失败: {str(e)}")
            return []
    
    def plan_route(
        self,
        origin_address: str,
        destination_address: str,
        origin_city: Optional[str] = None,
        destination_city: Optional[str] = None,
        route_type: str = "walking",
        origin_location: Optional[Dict[str, float]] = None,
        destination_location: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        规划路线
        
        Args:
            origin_address: 起点地址
            destination_address: 终点地址
            origin_city: 起点城市
            destination_city: 终点城市
            route_type: 路线类型 (walking/driving/transit)
            
        Returns:
            路线信息
        """
        try:
            direct_result = self._plan_route_by_location(
                origin_location=origin_location,
                destination_location=destination_location,
                city=origin_city or destination_city,
                route_type=route_type,
            )
            if direct_result:
                return direct_result

            # 根据路线类型选择工具
            tool_map = {
                "walking": "maps_direction_walking_by_address",
                "driving": "maps_direction_driving_by_address",
                "transit": "maps_direction_transit_integrated_by_address"
            }
            
            tool_name = tool_map.get(route_type, "maps_direction_walking_by_address")
            
            # 构建参数
            arguments = {
                "origin_address": origin_address,
                "destination_address": destination_address
            }
            
            # 公共交通需要城市参数
            if route_type == "transit":
                if origin_city:
                    arguments["origin_city"] = origin_city
                if destination_city:
                    arguments["destination_city"] = destination_city
            else:
                # 其他路线类型也可以提供城市参数提高准确性
                if origin_city:
                    arguments["origin_city"] = origin_city
                if destination_city:
                    arguments["destination_city"] = destination_city
            
            # 调用MCP工具
            result = self.mcp_tool.run({
                "action": "call_tool",
                "tool_name": tool_name,
                "arguments": arguments
            })
            
            print(f"路线规划结果: {result[:200]}...")
            parsed = self._parse_route_result(result, route_type)
            if parsed:
                return parsed
            return {}
            
        except Exception as e:
            print(f"❌ 路线规划失败: {str(e)}")
            return {}

    def _plan_route_by_location(
        self,
        origin_location: Optional[Dict[str, float]],
        destination_location: Optional[Dict[str, float]],
        city: Optional[str],
        route_type: str,
    ) -> Dict[str, Any]:
        if not origin_location or not destination_location:
            return {}
        origin = self._format_lng_lat(origin_location)
        destination = self._format_lng_lat(destination_location)
        if not origin or not destination:
            return {}

        if route_type == "walking":
            url = "https://restapi.amap.com/v3/direction/walking"
            params = {"key": self.settings.amap_api_key, "origin": origin, "destination": destination}
        elif route_type == "driving":
            url = "https://restapi.amap.com/v3/direction/driving"
            params = {"key": self.settings.amap_api_key, "origin": origin, "destination": destination, "extensions": "base"}
        else:
            if not city:
                return {}
            url = "https://restapi.amap.com/v3/direction/transit/integrated"
            params = {
                "key": self.settings.amap_api_key,
                "origin": origin,
                "destination": destination,
                "city": city,
                "cityd": city,
                "extensions": "all",
            }
        try:
            response = self.http_client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            print(f"❌ 高德路线HTTP查询失败: {exc}")
            log_event(
                "transport_route_http_error",
                {"route_type": route_type, "city": city or "", "error": repr(exc)},
            )
            return {}
        if str(data.get("status")) != "1":
            print(f"❌ 高德路线HTTP返回失败: {data.get('info') or data.get('infocode')}")
            log_event(
                "transport_route_http_error",
                {
                    "route_type": route_type,
                    "city": city or "",
                    "info": data.get("info") or "",
                    "infocode": data.get("infocode") or "",
                },
            )
            return {}
        if route_type == "transit":
            parsed = self._parse_amap_transit_response(data)
        else:
            parsed = self._parse_amap_path_response(data, route_type)
        if not parsed:
            log_event(
                "transport_route_http_empty",
                {
                    "route_type": route_type,
                    "city": city or "",
                    "raw_preview": self._preview_route_text(json.dumps(data, ensure_ascii=False)),
                },
            )
        return parsed

    def _format_lng_lat(self, location: Dict[str, float]) -> str:
        try:
            lng = float(location.get("longitude"))
            lat = float(location.get("latitude"))
        except Exception:
            return ""
        if abs(lng) < 0.000001 and abs(lat) < 0.000001:
            return ""
        return f"{lng:.6f},{lat:.6f}"

    def _parse_amap_path_response(self, data: Dict[str, Any], route_type: str) -> Dict[str, Any]:
        paths = ((data.get("route") or {}).get("paths") or [])
        if not paths:
            return {}
        path = paths[0]
        distance = self._coerce_route_number(path.get("distance")) or 0
        duration = self._coerce_route_number(path.get("duration")) or 0
        cost = self._coerce_route_number(path.get("tolls") or path.get("toll_distance"))
        steps = path.get("steps") or []
        description = "；".join(
            str(step.get("instruction") or "").strip()
            for step in steps[:8]
            if isinstance(step, dict) and str(step.get("instruction") or "").strip()
        )
        return {
            "distance": float(distance),
            "duration": int(duration),
            "cost": int(round(cost)) if cost else None,
            "route_type": route_type,
            "description": description,
            "raw": data,
        }

    def _parse_amap_transit_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        transits = ((data.get("route") or {}).get("transits") or [])
        if not transits:
            return {}
        parsed_options = []
        for transit in transits:
            if not isinstance(transit, dict):
                continue
            parsed = self._parse_single_amap_transit(transit, data)
            if parsed:
                parsed_options.append(parsed)
        if not parsed_options:
            return {}
        parsed_options.sort(
            key=lambda item: (
                0 if item.get("description") else 1,
                int(item.get("duration") or 999999999),
            )
        )
        return parsed_options[0]

    def _parse_single_amap_transit(self, transit: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        distance = self._coerce_route_number(transit.get("distance")) or self._sum_transit_distance(transit)
        duration = self._coerce_route_number(transit.get("duration")) or 0
        cost = self._coerce_route_number(transit.get("cost"))
        description = self._build_route_description(transit, "transit")
        if not distance and not duration:
            return {}
        return {
            "distance": float(distance or 0),
            "duration": int(duration or 0),
            "cost": int(round(cost)) if cost is not None else None,
            "route_type": "transit",
            "description": description,
            "raw": data,
        }

    def _sum_transit_distance(self, transit: Dict[str, Any]) -> float:
        total = 0.0
        for segment in transit.get("segments") or []:
            if not isinstance(segment, dict):
                continue
            total += self._first_number(segment, ("distance", "距离")) or 0
        return total

    def _parse_route_result(self, result: Any, route_type: str) -> Dict[str, Any]:
        """Parse Amap MCP route output into distance/duration/cost when possible."""
        if isinstance(result, dict):
            candidates = [result]
            text = json.dumps(result, ensure_ascii=False)
        else:
            text = str(result or "")
            candidates = self._extract_json_objects(text)

        for candidate in candidates:
            route = self._find_route_payload(candidate)
            if not route:
                continue
            distance = self._first_number(route, ("distance", "距离"))
            duration = self._first_number(route, ("duration", "time", "耗时", "时间"))
            cost = self._extract_route_cost(route, route_type)
            description = self._build_route_description(route, route_type)
            if distance is not None or duration is not None:
                return {
                    "distance": float(distance or 0),
                    "duration": int(duration or 0),
                    "cost": int(round(cost)) if cost is not None else None,
                    "route_type": route_type,
                    "description": description,
                    "raw": candidate,
                }

        distance = self._regex_number(text, (r"距离[:：]?\s*([\d.]+)\s*(公里|千米|km|米|m)", r"([\d.]+)\s*(公里|千米|km|米|m)"))
        duration = self._regex_duration(text)
        cost = self._regex_number(text, (r"(?:费用|票价|花费|价格|票费)[:：]?\s*(?:¥|￥)?\s*([\d.]+)\s*元?",))
        if distance is not None or duration is not None:
            return {
                "distance": float(distance or 0),
                "duration": int(duration or 0),
                "cost": int(round(cost)) if cost is not None else None,
                "route_type": route_type,
                "description": "",
                "raw": text,
            }
        return {}

    def _extract_json_objects(self, text: str) -> List[Dict[str, Any]]:
        objects: List[Dict[str, Any]] = []
        start = text.find("{")
        end = text.rfind("}")
        if 0 <= start < end:
            try:
                parsed = json.loads(text[start:end + 1])
                if isinstance(parsed, dict):
                    objects.append(parsed)
            except Exception:
                pass
        for match in re.finditer(r"\{.*?\}", text, re.DOTALL):
            try:
                parsed = json.loads(match.group())
            except Exception:
                continue
            if isinstance(parsed, dict):
                objects.append(parsed)
        return objects

    def _find_route_payload(self, value: Any) -> Optional[Dict[str, Any]]:
        if isinstance(value, dict):
            keys = {str(key).lower() for key in value.keys()}
            if keys & {"distance", "duration", "cost", "fee", "route", "paths", "transits"}:
                return value
            for child in value.values():
                found = self._find_route_payload(child)
                if found:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = self._find_route_payload(child)
                if found:
                    return found
        return None

    def _first_number(self, value: Any, names: tuple[str, ...]) -> Optional[float]:
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower() in names or str(key) in names:
                    number = self._coerce_route_number(item)
                    if number is not None:
                        return number
            for item in value.values():
                number = self._first_number(item, names)
                if number is not None:
                    return number
        elif isinstance(value, list):
            for item in value:
                number = self._first_number(item, names)
                if number is not None:
                    return number
        return None

    def _first_text(self, value: Any, names: tuple[str, ...]) -> str:
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower() in names or str(key) in names:
                    if isinstance(item, str):
                        return item
                    if isinstance(item, list):
                        return "；".join(str(part) for part in item[:5])
            for item in value.values():
                text = self._first_text(item, names)
                if text:
                    return text
        elif isinstance(value, list):
            for item in value:
                text = self._first_text(item, names)
                if text:
                    return text
        return ""

    def _extract_route_cost(self, route: Dict[str, Any], route_type: str) -> Optional[float]:
        if route_type == "transit":
            cost = self._first_number(route, ("cost", "price", "ticket_price", "bus_cost", "费用", "票价", "票费", "价格"))
            if cost is not None:
                return cost
            segment_prices = self._collect_numbers_by_key(route, ("price", "cost", "ticket_price", "费用", "票价", "票费"))
            if segment_prices:
                return max(segment_prices)
        if route_type == "driving":
            cost = self._first_number(route, ("tolls", "toll", "cost", "fee", "expense", "费用", "过路费"))
            if cost is not None:
                return cost
        return self._first_number(route, ("cost", "fee", "expense", "price", "费用", "票价", "花费", "价格"))

    def _collect_numbers_by_key(self, value: Any, names: tuple[str, ...]) -> List[float]:
        numbers: List[float] = []
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower() in names or str(key) in names:
                    number = self._coerce_route_number(item)
                    if number is not None:
                        numbers.append(number)
                numbers.extend(self._collect_numbers_by_key(item, names))
        elif isinstance(value, list):
            for item in value:
                numbers.extend(self._collect_numbers_by_key(item, names))
        return numbers

    def _build_route_description(self, route: Dict[str, Any], route_type: str) -> str:
        if route_type == "transit":
            transit_steps = self._collect_transit_steps(route)
            if transit_steps:
                return "；".join(transit_steps[:8])
        instruction_texts = self._collect_texts_by_key(
            route,
            ("instruction", "instructions", "description", "action", "assistant_action", "walk_type", "name"),
        )
        if instruction_texts:
            return "；".join(instruction_texts[:8])
        return self._first_text(route, ("description", "instruction", "instructions", "方案", "路径"))

    def _collect_transit_steps(self, value: Any) -> List[str]:
        steps: List[str] = []
        if isinstance(value, dict):
            buslines = value.get("buslines")
            if isinstance(buslines, list):
                for line in buslines:
                    if not isinstance(line, dict):
                        continue
                    line_name = str(line.get("name") or line.get("busline_name") or line.get("line_name") or "").strip()
                    departure = self._stop_name(line.get("departure_stop") or line.get("start_stop"))
                    arrival = self._stop_name(line.get("arrival_stop") or line.get("end_stop"))
                    if line_name:
                        if departure and arrival:
                            steps.append(f"乘坐{line_name}，{departure}上车，{arrival}下车")
                        else:
                            steps.append(f"乘坐{line_name}")
            railway = value.get("railway") or value.get("metro")
            if isinstance(railway, dict):
                name = str(railway.get("name") or railway.get("line") or "").strip()
                departure = self._stop_name(railway.get("departure_stop") or railway.get("start_stop"))
                arrival = self._stop_name(railway.get("arrival_stop") or railway.get("end_stop"))
                if name:
                    if departure and arrival:
                        steps.append(f"乘坐{name}，{departure}上车，{arrival}下车")
                    else:
                        steps.append(f"乘坐{name}")
            walking = value.get("walking")
            if isinstance(walking, dict):
                distance = self._first_number(walking, ("distance", "距离"))
                if distance and distance > 0:
                    steps.append(f"步行约{self._format_route_distance(distance)}")
            for item in value.values():
                steps.extend(self._collect_transit_steps(item))
        elif isinstance(value, list):
            for item in value:
                steps.extend(self._collect_transit_steps(item))
        return self._dedupe_texts(steps)

    def _collect_texts_by_key(self, value: Any, names: tuple[str, ...]) -> List[str]:
        texts: List[str] = []
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower() in names or str(key) in names:
                    if isinstance(item, str) and item.strip():
                        texts.append(item.strip())
                texts.extend(self._collect_texts_by_key(item, names))
        elif isinstance(value, list):
            for item in value:
                texts.extend(self._collect_texts_by_key(item, names))
        return self._dedupe_texts(texts)

    def _stop_name(self, value: Any) -> str:
        if isinstance(value, dict):
            return str(value.get("name") or value.get("station") or "").strip()
        return str(value or "").strip()

    def _dedupe_texts(self, values: List[str]) -> List[str]:
        result: List[str] = []
        seen: set[str] = set()
        for value in values:
            text = " ".join(str(value or "").split())
            if text and text not in seen:
                seen.add(text)
                result.append(text)
        return result

    def _format_route_distance(self, distance_m: float) -> str:
        if distance_m >= 1000:
            return f"{distance_m / 1000:.1f}公里"
        return f"{round(distance_m)}米"

    def _coerce_route_number(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value)
        match = re.search(r"[\d.]+", text)
        if not match:
            return None
        number = float(match.group())
        if any(unit in text.lower() for unit in ("公里", "千米", "km")):
            return number * 1000
        if "分钟" in text:
            return number * 60
        if "小时" in text:
            return number * 3600
        return number

    def _regex_number(self, text: str, patterns: tuple[str, ...]) -> Optional[float]:
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                continue
            number = float(match.group(1))
            unit = match.group(2).lower() if len(match.groups()) >= 2 and match.group(2) else ""
            if unit in {"公里", "千米", "km"}:
                return number * 1000
            return number
        return None

    def _regex_duration(self, text: str) -> Optional[float]:
        hour_match = re.search(r"([\d.]+)\s*(小时|h)", text, re.IGNORECASE)
        minute_match = re.search(r"([\d.]+)\s*(分钟|min)", text, re.IGNORECASE)
        if hour_match or minute_match:
            hours = float(hour_match.group(1)) if hour_match else 0.0
            minutes = float(minute_match.group(1)) if minute_match else 0.0
            return hours * 3600 + minutes * 60
        return None

    def _preview_route_text(self, text: str, limit: int = 160) -> str:
        return " ".join(str(text or "").split())[:limit]
    
    def geocode(self, address: str, city: Optional[str] = None) -> Optional[Location]:
        """
        地理编码(地址转坐标)

        Args:
            address: 地址
            city: 城市

        Returns:
            经纬度坐标
        """
        try:
            arguments = {"address": address}
            if city:
                arguments["city"] = city

            result = self.mcp_tool.run({
                "action": "call_tool",
                "tool_name": "maps_geo",
                "arguments": arguments
            })

            print(f"地理编码结果: {result[:200]}...")

            # TODO: 解析实际的坐标数据
            return None

        except Exception as e:
            print(f"❌ 地理编码失败: {str(e)}")
            return None

    def get_poi_detail(self, poi_id: str) -> Dict[str, Any]:
        """
        获取POI详情

        Args:
            poi_id: POI ID

        Returns:
            POI详情信息
        """
        try:
            result = self.mcp_tool.run({
                "action": "call_tool",
                "tool_name": "maps_search_detail",
                "arguments": {
                    "id": poi_id
                }
            })

            print(f"POI详情结果: {result[:200]}...")

            # 解析结果并提取图片
            import json
            import re

            # 尝试从结果中提取JSON
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data

            return {"raw": result}

        except Exception as e:
            print(f"❌ 获取POI详情失败: {str(e)}")
            return {}


# 创建全局服务实例
_amap_service = None


def get_amap_service() -> AmapService:
    """获取高德地图服务实例(单例模式)"""
    global _amap_service
    
    if _amap_service is None:
        _amap_service = AmapService()
    
    return _amap_service

