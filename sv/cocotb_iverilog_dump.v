module cocotb_iverilog_dump();
initial begin
    $dumpfile("CordicAccelerator.vcd");
    $dumpvars(0, CordicAccelerator);
end
endmodule
