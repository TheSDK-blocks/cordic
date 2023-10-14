import cocotb
from cocotb.triggers import ClockCycles, RisingEdge
from cocotb.clock import Clock


@cocotb.test()
async def test_CordicAccelerator(dut):
    cocotb.start_soon(Clock(dut.clock, period=2, units="step").start())
    dut.reset.value = 1
    await ClockCycles(dut.clock, 2)
    dut.reset.value = 0
    with open(cocotb.plusargs["infile"]) as infile:
        for row in infile:
            dut.io_in_bits_rs1.value = int(row)
            dut.io_in_valid.value = 1
            await RisingEdge(dut.clock)
    dut.io_in_valid.value = 0
    await ClockCycles(dut.clock, 20)
