"""Personas used to steer the style/framing of generated questions.

Framed around goals and problems rather than roles/skills, so that ragas'
persona-conditioned prompts lean toward natural troubleshooting-style
questions instead of textbook fact-lookup quizzes.
"""

from ragas.testset.persona import Persona

PERSONAS = [
    Persona(
        name="Marco",
        role_description=(
            "A ship's technician whose equipment is showing a malfunction or "
            "error/alarm code. He needs to figure out, from the manual, what "
            "is wrong and what steps to take to diagnose and fix it."
        ),
    ),
    Persona(
        name="Elena",
        role_description=(
            "A marine compliance surveyor who needs to check whether a "
            "measured or installed value (voltage, current, clearance, "
            "rating, certification, etc.) meets the parameters required by "
            "the manual before she can sign off on an installation."
        ),
    ),
    Persona(
        name="Luca",
        role_description=(
            "A new crew member who has to carry out a specific operational "
            "task with a piece of equipment he is not yet familiar with, and "
            "consults the manual to understand exactly what to do."
        ),
    ),
]
