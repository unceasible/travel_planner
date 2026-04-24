# Task Memory Template

---
task_id: "<task_id>"
user_id: "<user_id>"
nickname: "<nickname>"
status: "active"
created_at: "<ISO8601>"
updated_at: "<ISO8601>"
city: "<city>"
date_range: "<start_date> to <end_date>"
travel_days: 0
update_mode: "initial"
---

## Form Snapshot
```json
{
  "nickname": "",
  "city": "",
  "start_date": "",
  "end_date": "",
  "travel_days": 0,
  "transportation": "",
  "accommodation": "",
  "preferences": [],
  "free_text_input": ""
}
```

## Conversation Log
```json
[
  {
    "role": "user",
    "message": "",
    "timestamp": "<ISO8601>"
  },
  {
    "role": "assistant",
    "message": "",
    "timestamp": "<ISO8601>",
    "update_mode": "initial"
  }
]
```

## Current Plan
```json
{
  "city": "",
  "start_date": "",
  "end_date": "",
  "days": [],
  "weather_info": [],
  "overall_suggestions": "",
  "budget": {}
}
```

## Budget Ledger
```json
{
  "days": [],
  "total_attractions": 0,
  "total_hotels": 0,
  "total_meals": 0,
  "total_transportation": 0,
  "total": 0
}
```

## Reflection Log
```json
[
  {
    "timestamp": "<ISO8601>",
    "status": "ok",
    "notes": []
  }
]
```
