"""Verifier-owned execution isolation for BIA S03.

Private to the verifier tree. The verifier runs the whole comparator ensemble
and the submitted arm itself, in processes it spawns, over channels it created,
and measures the graded validation loss in a process into which no submitted
byte is ever loaded.
"""
