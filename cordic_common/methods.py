import os
import sys
from BitVector import BitVector
from math import modf, floor, sqrt
from bitstring import Bits

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if not (os.path.abspath(SCRIPT_DIR) in sys.path):
    sys.path.append(os.path.abspath(SCRIPT_DIR))


def calc_k(iters):
    from cordic_constants import hyperbolic_repeat_indices
    k = 1
    for i in range(0, iters):
        k *= sqrt(1 + pow(2, -2 * i))
        if i in hyperbolic_repeat_indices:
            k *= sqrt(1 + pow(2, -2 * i))
    return k


def to_fixed_point(value: float, mantissa_bits: int, float_bits: int):
    """
    Converts float value into fixed-point representation.
    Fixed point is represented as tuple (BitVector, BitVector),
    where index 0 holds mantissa and index 1 holds fractional part
    """
    (frac, integer_signed) = modf(value)
    sign = (value < 0)
    integer = int(abs(integer_signed))
    frac_bits = floor((1 << float_bits) * abs(frac))
    bits = (integer << float_bits) | frac_bits
    if sign:
        bits *= -1
    bit_str = Bits(int=bits, length=(mantissa_bits + float_bits))
    # requires bitstring 4.1.0
    bit_vec = BitVector(bitstring=bit_str.tobitarray().to01())
    return bit_vec


def to_double(bit_vector: (BitVector, BitVector)):
    """
    Converts fixed-point value, represented by tuple (BitVector, BitVector),
    where index 0 holds mantissa and index 1 fractional,
    into float.
    """
    bt_vec = bit_vector[0] + bit_vector[1]
    return to_double_single(bt_vec, bit_vector[0].length(), bit_vector[1].length())


def to_double_single(bit_vector: BitVector, mantissa_bits: int, float_bits: int):
    """
    Converts fixed-point value, represented by BitVector, into float.
    Assumed that mantissa is located in bits 0:mantissa_bits, and
    fractional is located at mantissa_bits:(mantissa_bits+float_bits)
    """
    if bit_vector[0]:
        bt_vec = BitVector(
            intVal=((~bit_vector).int_val() + 1), size=(mantissa_bits + float_bits)
        )
        sign = -1
    else:
        bt_vec = bit_vector
        sign = 1
    integer = bt_vec[0:mantissa_bits].int_val()
    frac = bt_vec[mantissa_bits : (mantissa_bits + float_bits)].int_val() / (
        1 << float_bits
    )
    return sign * (integer + frac)
