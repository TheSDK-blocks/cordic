from BitVector import BitVector


class ripple_carry_adder:
    def __init__(self, bits: int, endianness: str = "big"):
        self.bits = bits
        assert endianness in [
            "big",
            "little",
        ], "Endianness must be either 'big' or 'little'"
        self.endianness = endianness

    def add(self, input_a: BitVector, input_b: BitVector):
        if self.endianness == "big":
            dir = -1
            offs = 1
        else:
            dir = 1
            offs = 0

        S_vec = self.bits * [0]
        C_vec = (self.bits + 1) * [0]
        for i in range(0, self.bits):
            S_vec[i] = input_a[dir * i - offs] ^ input_b[dir * i - offs] ^ C_vec[i]
            C_vec[i + 1] = (input_a[dir * i - offs] & input_b[dir * i - offs]) + (
                C_vec[i] & (input_a[dir * i - offs] ^ input_b[dir * i - offs])
            )
        if self.endianness == "big":
            S_vec.reverse()
        return BitVector(bitlist=S_vec)
