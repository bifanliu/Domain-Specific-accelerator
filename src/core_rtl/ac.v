`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: NYCU
// Engineer: Department of Mechanical Engineering
// 
// Create Date: 2023/12/16 22:58:19
// Design Name: Bi-Fan Liu
// Project Name: Acceleration Circuit(software transfer to hardware)
// Target Devices: FPGA arty-7-100T
// Tool Versions: vivado 2023.1
// Description: We use a 3-layer 784-48-10 MLP neural network model and this
//              project converts software calculations into hardware calculations
// 
// Dependencies: 32 bits Aqila core 
// 
//////////////////////////////////////////////////////////////////////////////////

`include "aquila_config.vh"

module ac #
(parameter XLEN = 32,
  parameter CLSIZE = `CLP    // Cache line size.
)
(
    /////////// System signals   ///////////////////////////////////////////////
    input                     clk_i, rst_i,
    
    // ac to writback
    input                     wribk_strobe_i,  // writeback trigger acceleration circuit
    input  [4 : 0]            addr_i,          // register file addr
    input  [XLEN-1 : 0]       data_i,          // register file data
    
    output                    actowribk_o,     // if acceleration will over than go back to writeback signal
    output [XLEN-1 : 0]       data_o,          // acceleration circuit to writeback data
    
    // ac to dcache
    output reg                a_strobe_o,      // trigger Dcache to find memory
    output                    a_rw_o,          // 0 is read, 1 is write
    output reg [XLEN-1  : 0]  a_addr_o,        // memory address
    output reg [XLEN-1 : 0]   a_data_o,        // if write back to memory than output data to Dcache
   
    input  [1 : 0]            a_offset_i,      // make sure correct line offset in dcache
    input  [CLSIZE-1 : 0]     a_data_i,        // data input form Dcache
    input                     a_ready_i        // Dcache is ready to input data or receive data
   

);

/* Neural Network struct paremeters
struct __NeuroNet
{
    float *neurons;             // Array that stores all the neuron values.   This point address is in nn_data[0], real memory address is in 80212930
    float *weights;             // Array that store all the weights & biases. This point address is in nn_data[1], real memory address is in 80212934

    float **previous_neurons;   // Pointers to the previous-layer neurons.    This point address is in nn_data[2], real memory address is in 80212938
    float **forward_weights;    // Pointers to the weights & bias.            This point address is in nn_data[3], real memory address is in 8021293c

    int n_neurons[MAX_LAYERS];  // The # of neurons in each layer.            This point address is in nn_data[4~11], real memory address is in 80212940
    int total_layers;           // The total # of layers.                     This value address is in nn_data[12], real memory address is in 80212960
    int total_neurons;          // The total # of neurons.                    This value address is in nn_data[13], real memory address is in 80212964
    int total_weights;          // The total # of weights.                    This value address is in nn_data[14], real memory address is in 80212968
    float *output;              // Pointer to the neurons of the output layer.This value address is in nn_data[15], real memory address is in 8021296c
} NeuroNet;
*/

integer i, j, i2, j2, i3;

(* mark_debug = "true" *) reg [XLEN-1 : 0] nn_addr; // save struct NeuroNet Dcache address in register
reg [XLEN-1 : 0] nn_data [15 : 0]; // save struct NeuroNet paremeters Dcache address in register
(* mark_debug = "true" *) reg [4 : 0] nn_ctr; // record where struct NeuroNet paremeters read in which register

(* mark_debug = "true" *) reg [XLEN-1 : 0] weight_addr;// record forward_weight[neuron_idx] Dcache address
(* mark_debug = "true" *) reg [XLEN-1 : 0] neurons_addr;// record previous_neurons[neuron_idx] Dcache address

// signal to call dcache
reg a_strobe_r;
reg a_rw_r;
reg a_ready_r;
(* mark_debug = "true" *) reg [CLSIZE-1 : 0] a_data_r;

wire a_strobe;

// paremeter from Forward computations
(* mark_debug = "true" *) reg  [XLEN-1 : 0] neuron_idx, layer_idx, idx, jdx;
wire [XLEN-1 : 0] inner_product;

// give floating-point ip signal      
(* mark_debug = "true" *) reg             s_axis_a_tvalid;                
(* mark_debug = "true" *) wire             s_axis_a_tready;               
(* mark_debug = "true" *) reg  [31 : 0]   s_axis_a_tdata;        
(* mark_debug = "true" *) reg             s_axis_b_tvalid;                
(* mark_debug = "true" *) wire             s_axis_b_tready;               
(* mark_debug = "true" *) reg  [31 : 0]   s_axis_b_tdata;
(* mark_debug = "true" *) reg             s_axis_c_tvalid;                
(* mark_debug = "true" *) wire             s_axis_c_tready;               
(* mark_debug = "true" *) reg [31 : 0]    s_axis_c_tdata;          
(* mark_debug = "true" *) reg             s_axis_operation_tvalid;        
(* mark_debug = "true" *) wire            s_axis_operation_tready;       
(* mark_debug = "true" *) reg  [7 : 0]    s_axis_operation_tdata; // 8'b0000_0000 -> Add , 8'b0000_0001 -> Sub
(* mark_debug = "true" *) wire            m_axis_result_tvalid;          
(* mark_debug = "true" *) reg             m_axis_result_tready;           
(* mark_debug = "true" *) wire [31 : 0]   m_axis_result_tdata;

// give floating-point compare ip signal      
reg                 s_compare_a_tvalid;                
wire                s_compare_a_tready;               
reg  [31 : 0]       s_compare_a_tdata;        
reg                 s_compare_b_tvalid;                
wire                s_compare_b_tready;               
reg  [31 : 0]       s_compare_b_tdata;      
wire                m_compare_result_tvalid;          
reg                 m_compare_result_tready;           
wire                m_compare_result_tdata;


//=======================================================
// Acceleration Circuit Finite State Machine
//=======================================================
localparam Init                 = 11,
           Analysis             = 12,
           RdNeuron             = 13,
           RdNeuronFinish       = 14,
           RdWeight             = 15,
           RdWeightFinish       = 16,
           RdNeurons            = 17,
           RdNeuronsFinish      = 18,
           RdNeuronAddr         = 19,
           RdNeuronAddrFinish   = 20,
           RdWeightAddr         = 21,
           RdWeightAddrFinish   = 22,
           Start                = 23,
           WriteToDcache        = 24,
           WriteFinish          = 25,
           Compare              = 26,
           CompareFinish        = 27;
           

// AC controller state registers
(* mark_debug = "true" *) reg [ 4 : 0] S, S_nxt;

// whether nn_addr is read into AC 
(* mark_debug = "true" *) wire nn_addr_read;

reg [XLEN-1 : 0] p_weight_b [3 : 0];// p_weight_b is p_weight buffer
reg [XLEN-1 : 0] p_neuron_b [3 : 0];// p_neuron_b is p_neuron buffer
(* mark_debug = "true" *) reg [XLEN-1 : 0] p_weight [3 : 0];// if p_weight_b is full and if w_p_ctr is 4 than p_weight_b put into p_weight
(* mark_debug = "true" *) reg [XLEN-1 : 0] p_neuron [3 : 0];// if p_neuron_b is full and if n_p_ctr is 4 than p_neuron_b put into p_weight
(* mark_debug = "true" *) reg [2 : 0] n_p_ctr;
(* mark_debug = "true" *) reg [2 : 0] w_p_ctr;
(* mark_debug = "true" *) reg [2 : 0] n_p_ctr_b;
(* mark_debug = "true" *) reg [2 : 0] w_p_ctr_b;

reg [XLEN-1 : 0] max_r[3 : 0];
reg [XLEN-1 : 0] max_ctr;

(* mark_dubug = "true" *) reg  [2 : 0] done;// decide floating point compute or not
reg is_jdx_done;// if jdx < nn->n_neurons[layer_idx-1] than jdx done

wire is_a1;// if is register file a1 than read data into nn_addr

// for compare max file
(* mark_debug = "true" *) reg [XLEN-1 : 0] max_idx;
(* mark_debug = "true" *) reg [XLEN-1 : 0] max;
(* mark_debug = "true" *) reg [XLEN-1 : 0] max_index;
reg max_done;

assign is_a1 = (addr_i == 11);
assign nn_addr_read = (nn_addr == 0);

//====================================================
// AC Controller FSM
//====================================================
always @(posedge clk_i)
begin
    if (rst_i)
        S <= Init;
    else
        S <= S_nxt;
end

always @(*) begin
    case (S)
        Init: 
            if (wribk_strobe_i)
                S_nxt = Analysis;
            else 
                S_nxt = Init;
        Analysis:
            if(!nn_addr_read)
                S_nxt = RdNeuron;
            else
                S_nxt = Analysis;
        RdNeuron:
            if (a_ready_i)
                S_nxt = RdNeuronFinish;
            else
                S_nxt = RdNeuron;
        RdNeuronFinish:
            if(nn_ctr < 12)
                S_nxt = RdNeuron;
            else
                S_nxt = RdWeightAddr;
        RdWeight:
            if (a_ready_i)
                S_nxt = RdWeightFinish;
            else
                S_nxt = RdWeight;
        RdWeightFinish:
            S_nxt = Start;
        RdNeurons:
            if (a_ready_i)
                S_nxt = RdNeuronsFinish;
            else
                S_nxt = RdNeurons;
        RdNeuronsFinish:
                S_nxt = Start;
        RdNeuronAddr:
            if (a_ready_i)
                S_nxt = RdNeuronAddrFinish;
            else
                S_nxt = RdNeuronAddr;
        RdNeuronAddrFinish:
            S_nxt = RdWeight;
        RdWeightAddr:
            if (a_ready_i)
                S_nxt = RdWeightAddrFinish;
            else
                S_nxt = RdWeightAddr;
        RdWeightAddrFinish:
            S_nxt = RdNeuronAddr;
        Start:
            if(is_jdx_done)
                S_nxt = WriteToDcache;
            else if(w_p_ctr_b == 4)
                S_nxt = RdWeight;
            else if(n_p_ctr_b == 4)
                S_nxt = RdNeurons;
            else
                S_nxt = Start;
        WriteToDcache:
            if(a_ready_i)
                S_nxt = WriteFinish;
            else
                S_nxt = WriteToDcache;
        WriteFinish:
            if(layer_idx >= nn_data[12]) // nn_data[12] is total layer
                S_nxt = Compare;
            else
                S_nxt = RdWeightAddr;
        Compare:
            if(a_ready_i)
                S_nxt = CompareFinish;
            else
                S_nxt = Compare;
        CompareFinish:
            if(max_idx == nn_data[6] && max_done)
                S_nxt = Init;
            else if(max_ctr == 4)
                S_nxt = Compare;
            else
                S_nxt = CompareFinish;
    endcase
end

// for jdx done signal control
always @(posedge clk_i) begin
    if(rst_i)
        is_jdx_done <= 0;
    else if(jdx >= (nn_data[3+layer_idx]+1) && m_axis_result_tvalid)
        is_jdx_done <= 1;
    else if(is_jdx_done && S == WriteToDcache)
        is_jdx_done <= 0;
end

// for neuron_idx signal control
always @(posedge clk_i) begin
    if(rst_i)
        neuron_idx <= 0;
    else if(S == RdNeuronFinish)
        neuron_idx <= nn_data[4];// nn_data[4] == nn->n_neurons[0]
    else if(S == Start && S_nxt == WriteToDcache)
        neuron_idx <= neuron_idx + 1;
end

// for idx signal control
always @(posedge clk_i) begin
    if(rst_i)
        idx <= {32{1'b0}};
    else if(S == Start && S_nxt == WriteToDcache)
        idx <= idx + 1;
    else if(idx >= nn_data[4+layer_idx] && S > RdNeuronFinish)
        idx <= 0;
end

// for layer_idx signal control
always @(posedge clk_i) begin
    if(rst_i)
        layer_idx <= 0;
    else if(S <= RdNeuronFinish)
        layer_idx <= 1;
    else if(idx >= nn_data[4+layer_idx] && S > RdNeuronFinish)// idx and layer_idx is Í¬²½
        layer_idx <= layer_idx + 1;
end

// for a_rw_r signal control
always @(posedge clk_i) begin
    if(rst_i)
        a_rw_r <= 0;
     else if(S == WriteToDcache)
        a_rw_r <= 1;
     else
        a_rw_r <= 0;
end

// for p_weight_b cosider offset is important for system
always @(posedge clk_i) begin
    if(rst_i) begin
        for(i = 0; i < 4; i = i + 1)
            p_weight_b[i] <= {XLEN{1'b0}};
        weight_addr <= {XLEN{1'b0}};
    end
    else if(S == RdWeightFinish) begin
        p_weight_b[3] <= a_data_r[31 : 0];
        p_weight_b[2] <= a_data_r[63 : 32];
        p_weight_b[1] <= a_data_r[95 : 64];
        p_weight_b[0] <= a_data_r[127 : 96];
        weight_addr <= weight_addr + (4-a_offset_i)*4;// next read address
    end
    else if(S == RdWeightAddrFinish) begin
        case (a_offset_i)
            2'b11: weight_addr <= a_data_r[ 31: 0];     // [127: 96]
            2'b10: weight_addr <= a_data_r[ 63: 32];    // [ 95: 64]
            2'b01: weight_addr <= a_data_r[ 95: 64];    // [ 63: 32]
            2'b00: weight_addr <= a_data_r[127: 96];    // [ 31:  0]
        endcase
    end
end

// for p_neuron_b control
always @(posedge clk_i) begin
    if(rst_i) begin
        for(j = 0; j < 4; j = j + 1)
            p_neuron_b[j] <= {XLEN{1'b0}};
        neurons_addr <= {XLEN{1'b0}};
    end
    else if(S == RdNeuronAddrFinish) begin
        case (a_offset_i)
            2'b11: neurons_addr <= a_data_r[ 31: 0];     // [127: 96]
            2'b10: neurons_addr <= a_data_r[ 63: 32];    // [ 95: 64]
            2'b01: neurons_addr <= a_data_r[ 95: 64];    // [ 63: 32]
            2'b00: neurons_addr <= a_data_r[127: 96];    // [ 31:  0]
        endcase
    end
    else if(S == RdNeuronsFinish) begin
        p_neuron_b[3] <= a_data_r[31 : 0];
        p_neuron_b[2] <= a_data_r[63 : 32];
        p_neuron_b[1] <= a_data_r[95 : 64];
        p_neuron_b[0] <= a_data_r[127 : 96];
        neurons_addr <= neurons_addr + (4-a_offset_i)*4;
    end
end

// Rd nn address from writeback
always @(posedge clk_i) begin
    if(rst_i || S == CompareFinish)
        nn_addr <= 0;
    else if(nn_addr_read && is_a1 && (S == Analysis || (S == Init && S_nxt == Analysis)))
        nn_addr <= data_i;
    else
        nn_addr <= nn_addr;
end

// Register input data from the main memory.
always @(*) begin
    if (a_ready_i)
        a_data_r = a_data_i;
    else
        a_data_r = a_data_r;
end

// for nn_ctr control
always @(posedge clk_i) begin
    if(rst_i) begin
        nn_ctr <= 0;
        for(j2 = 0; j2 < 16; j2 = j2 + 1)
            nn_data[j2] <= {XLEN{1'b0}};
    end
    else if(S == Analysis)
        nn_ctr <= 0;
    else if(S == RdNeuronFinish) begin
        nn_data[nn_ctr+3] <= a_data_r[31 : 0];
        nn_data[nn_ctr+2] <= a_data_r[63 : 32];
        nn_data[nn_ctr+1] <= a_data_r[95 : 64];
        nn_data[nn_ctr+0] <= a_data_r[127 : 96];
        nn_ctr <= nn_ctr + 4;
    end
    else if(S == Compare && S_nxt == CompareFinish)
        nn_data[15] <= nn_data[15] + (4-a_offset_i)*4;
end

// for a_addr_o control
always @(posedge clk_i) begin
    if(rst_i)
        a_addr_o <= 0;
    else begin
        case(S)
            RdNeuron:
                a_addr_o <= nn_addr + nn_ctr * 4;
            RdWeightAddr:
                a_addr_o <= nn_data[3] + neuron_idx * 4;
            RdNeuronAddr:
                a_addr_o <= nn_data[2] + neuron_idx * 4;
            RdWeight:
                a_addr_o <= weight_addr;
            RdNeurons:
                a_addr_o <= neurons_addr;
            WriteToDcache:
                a_addr_o <= nn_data[0] + (neuron_idx-1) * 4;// nn_data[0] = nn->neurons[0]
            Compare:
                a_addr_o <= nn_data[15];
            default:
                a_addr_o <= {XLEN{1'b0}};
        endcase
    end
end

// for a_strobe_o control
assign a_strobe = (S == RdNeuron || S == RdWeight || S == RdNeurons ||
                   S == RdWeightAddr || S == RdNeuronAddr || S == WriteToDcache || 
                   S == Compare);

always @(posedge clk_i)
    a_strobe_r <= a_strobe;

always @(posedge clk_i) begin
    if(S == rst_i)
        a_strobe_o <= 0;
    else if(!a_strobe_r && a_strobe)
        a_strobe_o <= 1;
    else
        a_strobe_o <= 0;
end

// for p_weight and p_neuron control
always @(posedge clk_i) begin
    if(rst_i) begin
        for(i2 = 0; i2 < 4; i2 = i2 + 1) begin
            p_weight[i2] <= {XLEN{1'b0}};
            p_neuron[i2] <= {XLEN{1'b0}};
        end
    end
    else begin
        if(w_p_ctr_b != 4 && w_p_ctr == 4) begin
            for(i2 = 0; i2 < 4; i2 = i2 + 1)
                p_weight[i2] <= p_weight_b[i2];
        end
        if(n_p_ctr_b != 4 && n_p_ctr == 4) begin
            for(i2 = 0; i2 < 4; i2 = i2 + 1)
                p_neuron[i2] <= p_neuron_b[i2];
        end
    end
end

// for done
always @(posedge clk_i) begin
    if(rst_i || S == WriteToDcache || S == Analysis)
        done <= 2'b00;
    else if(m_axis_result_tvalid && done == 1 && jdx <= (nn_data[3+layer_idx]))
        done <= 2;
     else if((done == 2 || done == 0) && n_p_ctr < 4 && w_p_ctr < 4 )
        done <= 1;
end

always @(posedge clk_i) begin
    if(rst_i || S == WriteToDcache || S == Analysis) begin
        w_p_ctr_b <= 4;
        n_p_ctr_b <= 4;
        w_p_ctr <= 4;
        n_p_ctr <= 4;
        jdx <= 0;
    end
    else if(S == RdWeightFinish)
        w_p_ctr_b <= a_offset_i;
    else if(S == RdNeuronsFinish)
        n_p_ctr_b <= a_offset_i;
    else begin
        if(w_p_ctr == 4 && w_p_ctr_b != 4) begin
            w_p_ctr <= w_p_ctr_b;
            w_p_ctr_b <= 4;// start read weight
        end
        if(n_p_ctr == 4 && n_p_ctr_b != 4) begin
            n_p_ctr <= n_p_ctr_b;
            n_p_ctr_b <= 4;
        end
    end
    if((done == 2 || done == 0) && n_p_ctr < 4 && w_p_ctr < 4) begin
        w_p_ctr <= w_p_ctr + 1;
        n_p_ctr <= n_p_ctr + 1;
        jdx <= jdx + 1;
    end
end

// for max ctr
always @(posedge clk_i) begin
    if(rst_i || S == WriteToDcache || S == Analysis)
        max_ctr <= 4;
    else if(S == Compare && S_nxt == CompareFinish)
        max_ctr <= a_offset_i;
    else if(S == CompareFinish && max_ctr < 4 && (max_idx == 0 || max_done) && max_idx < nn_data[6])
        max_ctr <= max_ctr + 1;
end

// for floating point 0 signal control
always @(posedge clk_i) begin
    if(rst_i) begin
        s_axis_operation_tdata = 8'b0000_0000;// add
        s_axis_a_tvalid <= 0;
        s_axis_b_tvalid <= 0;
        s_axis_c_tvalid <= 0;
        s_axis_operation_tvalid <= 1;
        m_axis_result_tready <= 1;
        s_axis_a_tdata <= {XLEN{1'b0}};
        s_axis_b_tdata <= {XLEN{1'b0}};
    end
    else if((done == 2 || done == 0) && n_p_ctr < 4 && w_p_ctr < 4) begin
        s_axis_a_tvalid <= 1;
        s_axis_b_tvalid <= 1;
        s_axis_c_tvalid <= 1;
        s_axis_a_tdata <= jdx == nn_data[3+layer_idx] ? 32'h3f800000 : p_neuron[n_p_ctr];
        s_axis_b_tdata <= p_weight[w_p_ctr];
    end
    else begin
        s_axis_a_tvalid <= 0;
        s_axis_b_tvalid <= 0;
        s_axis_c_tvalid <= 0;
    end
end

// for s_compare_a and s_compare_b
always @(posedge clk_i) begin
    if(rst_i) begin
        s_compare_a_tvalid <= 0;
        s_compare_b_tvalid <= 0;
        s_compare_a_tdata <= {XLEN{1'b0}};
        s_compare_b_tdata <= {XLEN{1'b0}};
        m_compare_result_tready <= 1;
    end
    else if(S == CompareFinish && max_ctr < 4 && (max_idx == 0 || max_done) && max_idx < nn_data[6]) begin
        s_compare_a_tvalid <= 1;
        s_compare_b_tvalid <= 1;
        s_compare_a_tdata <= max_r[max_ctr];
        s_compare_b_tdata <= max;
    end
    else begin
        s_compare_a_tvalid <= 0;
        s_compare_b_tvalid <= 0;
    end
end

// for a_data_o control
always @(posedge clk_i) begin
    if(rst_i)
        a_data_o <= {XLEN{1'b0}};
    else if(S == WriteToDcache)
        a_data_o <= s_axis_c_tdata[31] == 1 ? 0 : s_axis_c_tdata;
end

// for s_axis_c_tdata control
always @(posedge clk_i) begin
    if(rst_i)
        s_axis_c_tdata <= {XLEN{1'b0}};
    else if(S == RdWeightAddr)
        s_axis_c_tdata <= {XLEN{1'b0}};
    else if(m_axis_result_tvalid)
        s_axis_c_tdata <= m_axis_result_tdata;
end

// for max
always @(posedge clk_i) begin
    if(rst_i || S == Analysis)
        max <= 32'hbf800000;
    else if(m_compare_result_tvalid && m_compare_result_tdata)// result == 1 means a > b(max)
        max <= s_compare_a_tdata;
end

// for max_idx
always @(posedge clk_i) begin
    if(rst_i || S == Analysis)
        max_idx <= {XLEN{1'b0}};
    else if(S == CompareFinish && max_ctr < 4 && (max_idx == 0 || max_done))
        max_idx <= max_idx + 1;
end

// for max_index
always @(posedge clk_i) begin
    if(rst_i || S == Analysis)
        max_index <= 0;
    else if(m_compare_result_tvalid && m_compare_result_tdata)// result == 1 means a > b(max)
        max_index <= max_idx;
end

// for max done
always @(posedge clk_i) begin
    if(rst_i || S == Analysis)
        max_done <= 0;
    else if(m_compare_result_tvalid)
        max_done <= 1;
    else if(S == CompareFinish && max_ctr < 4 && (max_idx == 0 || max_done))
        max_done <= 0;
end

// for max_r
always @(posedge clk_i) begin
    if(rst_i) begin
        for(i3 = 0; i3 < 4; i3 = i3 + 1)
            max_r[i3] <= {XLEN{1'b0}};
    end
    else if(S == Compare && S_nxt == CompareFinish) begin
        max_r[3] <= a_data_r[31 : 0];
        max_r[2] <= a_data_r[63 : 32];
        max_r[1] <= a_data_r[95 : 64];
        max_r[0] <= a_data_r[127 : 96];
    end
end

//----------- Begin Cut here for INSTANTIATION Template ---// INST_TAG
floating_point_0 float_compute(
  .aclk(clk_i),                                         // input wire aclk
  .s_axis_a_tvalid(s_axis_a_tvalid),                    // input wire s_axis_a_tvalid
  .s_axis_a_tready(s_axis_a_tready),                    // output wire s_axis_a_tready
  .s_axis_a_tdata(s_axis_a_tdata),                      // input wire [31 : 0] s_axis_a_tdata
  .s_axis_b_tvalid(s_axis_b_tvalid),                    // input wire s_axis_b_tvalid
  .s_axis_b_tready(s_axis_b_tready),                    // output wire s_axis_b_tready
  .s_axis_b_tdata(s_axis_b_tdata),                      // input wire [31 : 0] s_axis_b_tdata
  .s_axis_c_tvalid(s_axis_c_tvalid),                    // input wire s_axis_c_tvalid
  .s_axis_c_tready(s_axis_c_tready),                    // output wire s_axis_c_tready
  .s_axis_c_tdata(s_axis_c_tdata),                      // input wire [31 : 0] s_axis_c_tdata
  .s_axis_operation_tvalid(s_axis_operation_tvalid),    // input wire s_axis_operation_tvalid
  .s_axis_operation_tready(s_axis_operation_tready),    // output wire s_axis_operation_tready
  .s_axis_operation_tdata(s_axis_operation_tdata),      // input wire [7 : 0] s_axis_operation_tdata
  .m_axis_result_tvalid(m_axis_result_tvalid),          // output wire m_axis_result_tvalid
  .m_axis_result_tready(m_axis_result_tready),          // input wire m_axis_result_tready
  .m_axis_result_tdata(m_axis_result_tdata)             // output wire [31 : 0] m_axis_result_tdata
);

floating_point_1 compare(
  .aclk(clk_i),
  .s_axis_a_tvalid(s_compare_a_tvalid),
  .s_axis_a_tready(s_compare_a_tready),
  .s_axis_a_tdata(s_compare_a_tdata),
  .s_axis_b_tvalid(s_compare_b_tvalid),
  .s_axis_b_tready(s_compare_b_tready),
  .s_axis_b_tdata(s_compare_b_tdata),
  .m_axis_result_tvalid(m_compare_result_tvalid),
  .m_axis_result_tready(m_compare_result_tready),
  .m_axis_result_tdata(m_compare_result_tdata)
);

// AC to writeback data connect
assign data_o = (S_nxt == Init && S == CompareFinish) ? max_index : 0;
assign a_rw_o = a_rw_r;
assign actowribk_o = (S_nxt == Init && S == CompareFinish) ? 1 : 0;

endmodule
