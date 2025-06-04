## Build a custom LLVM

Test for available compiler targets:

```script
llc  -march=riscv32 -mattr=help
```

## Testing for non-host architecture

Run with env:

```script
TRITON_GEN_TARGET=riscv32-unknown-linux TRITON_GEN_ARCH=rocket-rv32 python tutorials/03-matrix-multiply-cpu.py

or

TRITON_GEN_TARGET=aarch64-unknown-linux TRITON_GEN_ARCH=apple-m4 python tutorials/03-matrix-multiply-cpu.py
```

## Apple Metal compile stack (cnp from amd)

Compile down to LLVM IR, add `air` attributes, put all scalars into `constant buffer`.
Then use xcrun to generate binary.

```
xcrun -sdk macosx metal *.ll -o *.o
```

