add wave -radix decimal sim/:tb_cordic:cordic:io_*
add wave sim/:tb_cordic:cordic:clock
add wave sim/:tb_cordic:cordic:reset
add wave -radix decimal sim/:tb_cordic:cordic:cordicCore:inRegs_cordic_*
add wave -radix decimal sim/:tb_cordic:cordic:cordicCore:pipelineRegs_*

run -all


view wave
wave zoom full