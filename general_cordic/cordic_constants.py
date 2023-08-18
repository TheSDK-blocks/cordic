atan_lut = [
    0.7853981634,
    0.463647609,
    0.2449786631,
    0.1243549945,
    0.06241881,
    0.03123983343,
    0.01562372862,
    0.00781234106,
    0.003906230132,
]

atanh_lut = [
    None,  # +inf
    0.549306144334,
    0.255412811883,
    0.12565721414,
    0.062581571477,
    0.031260178491,
    0.015626271752,
    0.007812658952,
    0.003906269868,
]

# Hyperbolic mode does not converge without
# repeating some iterations
# Required repetitions can be calculated as 3k + 1 (k = 1,2,3 ...)
hyperbolic_repeat_indices = [4, 13, 40]