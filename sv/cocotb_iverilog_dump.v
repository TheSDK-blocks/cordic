module cocotb_iverilog_dump();
initial begin
    $dumpfile("cordic.vcd");
    $dumpvars(0, cordic);
end
endmodule
