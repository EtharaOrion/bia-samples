"""The frozen reference kernel, installed unchanged as the baseline control."""

import opstream


def run_block(tape, params, carry, start, length):
    return opstream.reference_run_block(tape, params, carry, start, length)
