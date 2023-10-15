import numpy as np
import cocotb
from cocotb.triggers import ClockCycles, RisingEdge
from cocotb.clock import Clock


@cocotb.test()
async def test_CordicAccelerator(dut):
    in_iofiles = cocotb.plusargs["in_iofiles"].split(",")
    in_ionames = cocotb.plusargs["in_ionames"].split(",")
    out_iofiles = cocotb.plusargs["out_iofiles"].split(",")
    out_ionames = cocotb.plusargs["out_ionames"].split(",")
    clock = getattr(dut, cocotb.plusargs["clock"])
    reset = getattr(dut, cocotb.plusargs["reset"])
    IOS = {}
    assert len(in_iofiles) == len(
        in_ionames
    ), "in_iofiles and in_ionames should be the same length!"
    assert len(out_iofiles) == len(
        out_ionames
    ), "out_iofiles and out_ionames should be the same length!"

    # Copy data from file into Python datatype
    for i in range(0, len(in_iofiles)):
        with open(in_iofiles[i], "r") as iofile:
            IOS[in_ionames[i]] = np.array([int(line.strip()) for line in iofile])

    # Run clock
    cocotb.start_soon(Clock(clock, period=2, units="step").start())

    # Reset
    reset.value = 1
    dut.io_in_valid.value = 0
    await ClockCycles(dut.clock, 2)
    reset.value = 0

    # Your testbench
    for i, sample in enumerate(IOS["io_in_bits_rs1"]):
        dut.io_in_bits_rs1.value = int(sample)
        dut.io_in_bits_op.value = int(IOS["io_in_bits_op"][i])
        dut.io_in_valid.value = 1
        await RisingEdge(clock)

    dut.io_in_valid.value = 0
    await ClockCycles(dut.clock, 20)
