add wave -radix decimal sim/:tb_cordic:cordic:io_*
add wave sim/:tb_cordic:cordic:clock
add wave sim/:tb_cordic:cordic:reset

run -all


view wave
wave zoom full