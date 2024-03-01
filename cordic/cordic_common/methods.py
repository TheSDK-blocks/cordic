import os
import sys
from BitVector import BitVector
from math import modf, floor, sqrt
from bitstring import Bits
from typing import Union
from numpy import int32, pi, float32, unpackbits

if not (os.path.dirname(os.path.abspath(__file__)) in sys.path):
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cordic_types import rotation_type


def calc_k(iters, rot_type: rotation_type):
    from cordic_constants import hyperbolic_repeat_indices

    i_init = 0 if rot_type == rotation_type.CIRCULAR else 1
    k = 1
    for i in range(i_init, iters):
        sqrtee = (
            1 + pow(2, -2 * i)
            if rot_type == rotation_type.CIRCULAR
            else 1 - pow(2, -2 * i)
        )
        k *= sqrt(sqrtee)
        if i in hyperbolic_repeat_indices:
            k *= sqrt(sqrtee)
    return k


def to_fixed_point(
    value: Union[float, float32, int, int32], mantissa_bits: int, float_bits: int, repr: str,
    ret_type: str = "BitVector"
):
    """
    Converts float or int value into fixed-point representation.
    Float values are translated to fixed-point before converting to BitVector.
    Int values are simply converted into a BitVector.
    """
    max_val = 2**mantissa_bits
    if isinstance(value, float) or isinstance(value, float32):
        if repr == "fixed-point":
            if ret_type == "BitVector":
                (frac, integer_signed) = modf(value)
                sign = value < 0
                integer = int(abs(integer_signed))
                frac_bits = floor((1 << float_bits) * abs(frac))
                bits = (integer << float_bits) | frac_bits
                if sign:
                    bits *= -1
                bit_str = Bits(int=bits, length=(mantissa_bits + float_bits))
                # requires bitstring 4.1.0
                bit_vec = BitVector(bitstring=bit_str.tobitarray().to01())
                return bit_vec
            elif ret_type == "numpy":
                return int32((value / max_val) * 2**32)
        elif repr == "pi":
            if ret_type == "BitVector":
                bits = int(((2 ** (mantissa_bits + float_bits - 1)) * value) / pi)
                bit_str = Bits(int=bits,
                            length=(mantissa_bits + float_bits))
                bit_vec = BitVector(bitstring=bit_str.tobitarray().to01())
                return bit_vec
            elif ret_type == "numpy":
                return int32(value / (2 * pi) * 2**32)
        else:
            raise ValueError("Undefined repr: " + repr)

    elif isinstance(value, int32):
        # requires bitstring 4.1.0
        if ret_type == "BitVector":
            bit_str = Bits(int=value, length=(mantissa_bits + float_bits))
            bit_vec = BitVector(bitstring=bit_str.tobitarray().to01())
            return bit_vec
        elif ret_type == "numpy":
            return value

    else:
        raise ValueError("Unsupported type: " + str(type(value)))


def to_double(bit_vector):
    """
    Converts fixed-point value, represented by tuple (BitVector, BitVector),
    where index 0 holds mantissa and index 1 fractional,
    into float.
    """
    bt_vec = bit_vector[0] + bit_vector[1]
    return to_double_single(bt_vec, bit_vector[0].length(), bit_vector[1].length())


def to_double_single(bit_vector, mantissa_bits: int, float_bits: int, repr: str):
    """
    Converts fixed-point value, represented by BitVector, into float.
    Assumed that mantissa is located in bits 0:mantissa_bits, and
    fractional is located at mantissa_bits:(mantissa_bits+float_bits)
    """
    if isinstance(bit_vector, BitVector):
        if bit_vector[0]:
            bt_vec = BitVector(
                intVal=((~bit_vector).int_val() + 1), size=(mantissa_bits + float_bits)
            )
            sign = -1
        else:
            bt_vec = bit_vector
            sign = 1

        if repr == "fixed-point":
            integer = bt_vec[0:mantissa_bits].int_val()
            frac = bt_vec[mantissa_bits: (mantissa_bits + float_bits)].int_val() / (
                1 << float_bits
            )
            return sign * (integer + frac)
        elif repr == "pi":
            intval = bt_vec.int_val()
            return sign * ((intval / 2**(mantissa_bits + float_bits - 1)) * pi)
    elif isinstance(bit_vector, int32):
        if repr == "fixed-point":
            max_val = 2**mantissa_bits
            return (bit_vector / 2**32) * max_val
        elif repr == "pi":
            return  (bit_vector / 2**32) * 2 * pi


