## Build a custom LLVM

Test for available compiler targets:

```script
llc  -march=riscv32 -mattr=help
```

## Testing for non-host architecture

Run with env:

```script
TRITON_GEN_TARGET=riscv32-unknown-linux TRITON_GEN_ARCH=rocket-rv32 python tutorials/03-matrix-multiply-cpu.py
```


