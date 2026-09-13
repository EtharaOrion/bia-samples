"""A fused first order recurrence for the frozen geometric-weight operator.

y[i] = ( r * y[i-1] + x[i] - (r ** K mod P) * x[i-K] ) mod P

Two Tape reads per output once the segment is K deep, against K reads for the reference.
"""


def run_block(tape, params, carry, start, length):
    modulus = int(params["modulus"])
    window = int(params["window"])
    ratio = int(params["ratio"])
    tail_weight = pow(ratio, window, modulus)

    accumulator = int(carry["accumulator"])
    history = [int(value) for value in carry["history"]]
    outputs = []

    for index in range(start, start + length):
        head = tape.at(index)
        source = index - window
        if source >= start:
            tail = tape.at(source)
        else:
            tail = history[0]
        accumulator = (ratio * accumulator + head - tail_weight * tail) % modulus
        outputs.append(accumulator)
        history = history[1:] + [head]

    return outputs, {"accumulator": accumulator, "history": history}
