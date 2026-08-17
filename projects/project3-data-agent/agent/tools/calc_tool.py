"""
calc_tool.py — Statistical calculation tool for healthcare metrics.
"""
from langchain.tools import tool
import json


@tool
def calculate_statistics(data_json: str) -> str:
    """
    Calculate healthcare statistics: rates, averages, percentages, trends.
    Input must be a JSON string with 'operation' and 'data' keys.

    Operations: 'rate', 'average', 'percentage', 'change'

    Examples:
    - Rate: {"operation": "rate", "data": {"numerator": 45, "denominator": 1200, "label": "readmission rate"}}
    - Average: {"operation": "average", "data": {"values": [3, 5, 7, 4], "label": "length of stay (days)"}}
    - Change: {"operation": "change", "data": {"old_value": 120, "new_value": 145, "label": "admissions"}}
    """
    try:
        params = json.loads(data_json)
        op = params["operation"]
        data = params["data"]

        if op == "rate":
            rate = (data["numerator"] / data["denominator"]) * 100
            return f"{data.get('label', 'Rate')}: {rate:.2f}% ({data['numerator']} / {data['denominator']})"

        elif op == "average":
            values = data["values"]
            avg = sum(values) / len(values)
            return f"Average {data.get('label', 'value')}: {avg:.2f} (min: {min(values)}, max: {max(values)}, n={len(values)})"

        elif op == "percentage":
            pct = (data["part"] / data["total"]) * 100
            return f"{data.get('label', 'Percentage')}: {pct:.1f}% ({data['part']} of {data['total']})"

        elif op == "change":
            change = data["new_value"] - data["old_value"]
            pct_change = (change / data["old_value"]) * 100
            direction = "increase" if change > 0 else "decrease"
            return (
                f"{data.get('label', 'Value')} {direction}: "
                f"{data['old_value']} → {data['new_value']} "
                f"({'+' if change > 0 else ''}{change}, {pct_change:+.1f}%)"
            )
        else:
            return f"Unknown operation: {op}. Use: rate, average, percentage, change"

    except (json.JSONDecodeError, KeyError, ZeroDivisionError) as e:
        return f"Calculation error: {str(e)}. Check your JSON input format."
