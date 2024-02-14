module cocotb_iverilog_dump();
initial begin
    $dumpfile("CordicTop.vcd");
    $dumpvars(0, CordicTop);
end
endmodule
