# Pipelined configurable CORDIC
For more hardware specs and configuration options, see `chisel/README.md`.

## IOS

- `io_in_valid` - Set this to 1 when feeding valid input data
- `io_in_ready` - Input ready to receive data
- `io_in_bits_rs1` - Input 1 (x)
- `io_in_bits_rs2` - Input 2 (y)
- `io_in_bits_rs3` - Input 3 (z)
- `io_in_bits_control` - Control bits - see `chisel/README.md`
- `io_out_bits_dOut` - Data out - see `chisel/README.md`
- `io_out_bits_cordic_x` - Output x
- `io_out_bits_cordic_y` - Output y
- `io_out_bits_cordic_z` - Output z
- `io_out_valid` - 1 when valid data out
- `io_out_ready` - Output ready to receive data
- `clock` - clock
- `reset` - reset

You can provide input values either in `numpy.float32` or `numpy.int32` format. `numpy.int32` will be transmitted to the module as is, but `numpy.float32` value is first converted into a fixed-point representation. See `methods.to_fixed_point` for details.

Output values are, by default, provided converted into `numpy.float32` values. However, this can be turned off by setting `self.convert_output` to `False`. Then, outputs are `numpy.int32` values.

## Example

In `__init__.py` file there is a self-test function that shows the basic functionality of the entity.

## Notes
In the code, there are mentios of using BitVector's. They are not used anymore with the newest model_3. This is because computing with them is very slow.

Furthermore, there is a flow that uses cocotb instead of TheSydeKick rtl entity. This is also experimental, and not used by default. The default flow uses TheSydeKick rtl entity.