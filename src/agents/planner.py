class PlannerAgent:
    """
    Breaks down the user query into a fixed sequence of subtasks.
    This defines the overall agentic flow.
    """

    def __init__(self):
        pass

    def plan(self, user_query):
        tasks = [
            {"task": "load_data", "description": "Load and summarize dataset"},
            {"task": "generate_insights", "description": "Create hypotheses explaining metric changes"},
            {"task": "validate_insights", "description": "Evaluate hypotheses using quantitative checks"},
            {"task": "generate_creatives", "description": "Suggest new creative directions for low-CTR campaigns"},
            {"task": "compile_report", "description": "Write final marketing insights report"}
        ]

        return {
            "query": user_query,
            "tasks": tasks
        }
