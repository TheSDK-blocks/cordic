from BitVector import BitVector


class adder_subtractor:
    def __init__(self, bits: int, endianness: str = "big"):
        self.bits = bits
        assert endianness in [
            "big",
            "little",
        ], "Endianness must be either 'big' or 'little'"
        self.endianness = endianness

    def add(self, input_a: BitVector, input_b: BitVector):
        return self._add(input_a, input_b, 0)

    def sub(self, input_a: BitVector, input_b: BitVector):
        return self._add(input_a, input_b, 1)

    def _add(self, input_a: BitVector, input_b: BitVector, D: int):
        """Adds/subtracts input_a from input_b.

        Args:
            input_a (BitVector): minuend
            input_b (BitVector): subtrahend
            D (int): perform addition (0) or subtraction (1)

        Returns:
            BitVector: calculation result
        """
        if self.endianness == "big":
            dir = -1
            offs = 1
        else:
            dir = 1
            offs = 0

        S_vec = self.bits * [0]
        C_vec = (self.bits + 1) * [0]
        C_vec[0] = D
        for i in range(0, self.bits):
            b = input_b[dir * i - offs] if D == 0 else input_b[dir * i - offs] ^ D
            S_vec[i] = b ^ input_a[dir * i - offs] ^ C_vec[i]
            C_vec[i + 1] = (b & input_a[dir * i - offs]) + (
                C_vec[i] & (b ^ input_a[dir * i - offs])
            )
        if self.endianness == "big":
            S_vec.reverse()
        return BitVector(bitlist=S_vec)
