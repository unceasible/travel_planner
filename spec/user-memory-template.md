# User Memory Template

---
user_id: "<user_id>"
nickname: "<nickname>"
created_at: "<ISO8601>"
updated_at: "<ISO8601>"
---

## Profile
```json
{
  "preferences": [],
  "dislikes": [],
  "constraints": [],
  "budget_sensitivity": "",
  "notes": [],
  "last_task_id": ""
}
```

## Update History
```json
[
  {
    "timestamp": "<ISO8601>",
    "source_task_id": "<task_id>",
    "source_message": "",
    "applied_changes": {
      "preferences": [],
      "dislikes": [],
      "constraints": [],
      "budget_sensitivity": "",
      "notes": []
    }
  }
]
```
