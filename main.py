"""Точка входу: запуск агента аналізу фінансових витрат."""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.agent import ExpenseAnalysisAgent


def main():
    agent = ExpenseAnalysisAgent()
    agent.run()


if __name__ == "__main__":
    main()
