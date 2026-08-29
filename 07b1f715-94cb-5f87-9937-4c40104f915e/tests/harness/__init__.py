"""Verifier-owned execution isolation for BIA S02.

Private to the verifier tree. The verifier uses this package to re-execute the
submitted update rule itself, in processes it spawns, over channels it created,
and to measure the graded loss in a process into which no submitted byte is
ever loaded.
"""
