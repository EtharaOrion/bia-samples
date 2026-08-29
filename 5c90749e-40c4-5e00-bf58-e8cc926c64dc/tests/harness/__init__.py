"""Verifier-owned execution isolation for BIA S01.

Nothing in this package is on the agent-visible surface. The verifier uses it
to re-execute the submission itself, in a process the verifier spawns, over a
channel the verifier owns, and to measure the graded quantity in a third
process into which no submitted byte is ever loaded.
"""
