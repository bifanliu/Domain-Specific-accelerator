# Domain-Specific accelerator
I have documented the process of setting up the environment for this class in detail, and the detailed information about DSA will be included in the course materials. I am using the Home edition of Windows 11

```material/.```
```
./material
├── DSA                     <===  Project Introduction
└── RISCV_ISA.pdf           <=== about RISC-V 
```

The neural network C program is located entirely under the directory ```ocr/.```
```
├── Makefile      
├── elibc                   <=== C lib       
├── data/
    ├── test-images.dat     <=== Data files to be copied to the SD card.
    ├── test-lavels.dat     <=== Data files to be copied to the SD card.
    └── weights.dat         <=== Data files to be copied to the SD card.
├── file_read.c             <=== The source code of the MLP model for OCR
├── file_read.h             <=== The source code of the MLP model for OCR
├── neuronet.c              <=== The source code of the MLP model for OCR 
├── neuronet.h              <=== The source code of the MLP model for OCR
├── ocr.c                   <=== The source code of the MLP model for OCR
├── ocr.ld                  <=== The source code of the MLP model for OCR
└── sdcard/
    ├── fat32.c             <=== SD card FAT32 file system I/O routines.
    ├── fat32.h             <=== SD card FAT32 file system I/O routines.
    ├── sd.c                <=== SD card FAT32 file system I/O routines.
    ├── sd.h                <=== SD card FAT32 file system I/O routines.
    ├── spi.c               <=== SD card FAT32 file system I/O routines.
    └── spi.h               <=== SD card FAT32 file system I/O routines.
```

## Set HW platform

1. download installer for Windows in: [Here](https://www.xilinx.com/support/download.html), you should have an account before downloading.
2. Excute download file ```Xilinx_Unified_2023.1_0507_1903_Win64```
3. you will see screen like mpd23_HW0_Simulation of a HW-SW Platform.pdf like page 6 ~ 8 following
4. Go to [Here](https://github.com/Digilent/vivado-boards), download the
directory ```./new/board_files/arty-a7-100/*```
5. Make a directory ```Digilent/``` under ```<INST_DIR>/Vivado/2023.1/
data/xhub/boards/XilinxBoardStore/boards/```
6. Put ```arty-a7-100/*``` under ```Digilent/```

7. Download ```src``` from project
8. Unzip the package to Desktop
9. In Unzip file excute cmd and enter ```<INST_DIR>\Vivado\2023.1\bin\vivado.bat -mode batch –source build.tcl```
```
INST_DIR ===> 看你將下載的xlinux放置在哪一個槽和file內,像我是放在D:\Xlinux,所以INST_DIR = D:\Xlinux
```
10. In upzip file you will see ```aquila_mpd``` click this file and excute ```aquila_mpd.xpr```

## Set SW platform

1. Download WSL+ and set c enviroment
2. Enter ```export PATH=$PATH:/opt/riscv/bin```
3. Enter ```export RISCV=/opt/riscv```
4. Enter ```make``` will generate .elf file
5. Use Tera Term to let .elf file input fpga

## Linker Script

Some Linker Example
``` 
__stack_size = 0x800;
__heap_size = 0x5000;
__heap_start = __stack_top + __heap_size;
MEMORY
{
    code_ram (rx!rw) : ORIGIN = 0x00000000, LENGTH = 0x5000
    data_ram (rw!x) : ORIGIN = 0x00005000, LENGTH = 0x4000
}
ENTRY(crt0)
SECTIONS
{
    .text :
    {
        libelibc.a(.text)
        *(.text)
    } > code_ram
    .data :
    {
        *(.data)
        *(.bss)
        *(.rodata*)
    } > data_ram
    .stack : ALIGN(0x10)
    {
        . += __stack_size;
        __stack_top = .;
    } > data_ram
}
```

1. Define Some important variable address and size
2. Set Entry Point in this HW0 our Entry Point is ```crt0```
3. Define Section content
* .bss: Not initial variable
* .text: code after compiling
* .data: Have initial variable

## Program Binary File Formats

* .mem – used for the initialization of the on-chip memory
* .elf – the standard UNIX Executable and Linkable Format
    
    * The ELF file path/name is defined in ```aquila_config.h```
    * uartboot loader will find .elf and accroding .elf content load to memory and uartboot find Entry point and jump to there

## Simulation Using Vivado Simulator

* soc_top.v – for circuit synthesis
* soc_tb.v – for circuit simulation
* printf()
    
    * At circuit level, printf() sends ASCIIs to the uart module
    * In simulation mode, the uart module will sent the ASCIIs to the “Tcl Console” of Vivado
    * There is a trap in uart.v such that when the ASCII code 0x03 is printed, the simulation will terminate

