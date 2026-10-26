import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import agent_logger, log_tool_call, EvaluationLogger

# Test agent logger
agent_logger.info("Test log entry")
print("✅ agent_logger OK")

# Test tool call logging
log_tool_call("sql_query_tool", "SELECT COUNT(*) FROM customers", "[{'count': 91}]", 2.3)
print("✅ log_tool_call OK")

# Test evaluation logger
eval_logger = EvaluationLogger()
eval_logger.log_result(
    question="How many customers?",
    expected="91",
    actual="There are 91 customers.",
    correct=True,
    response_time=3.2,
    model="claude-sonnet-4-6",
    tools_used=["sql_query_tool"],
)
print("✅ log_result OK")

summary = eval_logger.get_summary()
print("✅ get_summary OK:", summary)