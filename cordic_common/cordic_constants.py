from math import atan, atanh

iters = 30

atan_lut = [atan(2**-i) for i in range(0, iters)]

# atanh(1) is inf
atanh_lut = [None] + [atanh(2**-i) for i in range(1, iters)]

# Hyperbolic mode does not converge without
# repeating some iterations
# Required repetitions can be calculated as 3k + 1 (k = 1,2,3 ...)
hyperbolic_repeat_indices = [4, 13, 40]
