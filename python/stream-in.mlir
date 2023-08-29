#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @matmul_kernel_0d1d2d3d4d5d6d7c8d9c10d11c(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #triton_gpu.mfma<{nonKDim = 32, warpsPerCTA = [2, 2], isTransposed = false}>>
    %cst_0 = arith.constant dense<32> : tensor<128x32xi32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x32xf32, #blocked>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<32x128xf32, #blocked1>
    %c31_i32 = arith.constant 31 : i32
    %c127_i32 = arith.constant 127 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg4, %c127_i32 : i32
    %4 = arith.divsi %3, %c128_i32 : i32
    %5 = arith.muli %4, %c8_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.subi %2, %7 : i32
    %9 = "triton_gpu.cmpi"(%8, %c8_i32) <{predicate = 2 : i64}> : (i32, i32) -> i1
    %10 = arith.select %9, %8, %c8_i32 : i32
    %11 = arith.remsi %0, %10 : i32
    %12 = arith.addi %7, %11 : i32
    %13 = arith.remsi %0, %5 : i32
    %14 = arith.divsi %13, %10 : i32
    %15 = arith.muli %12, %c128_i32 : i32
    %16 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %17 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %18 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %19 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %20 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %21 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %22 = tt.splat %15 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %23 = tt.splat %15 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %24 = tt.splat %15 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %25 = arith.addi %22, %16 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %26 = arith.addi %23, %18 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %27 = arith.addi %24, %20 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %28 = arith.muli %14, %c128_i32 : i32
    %29 = tt.splat %28 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %30 = tt.splat %28 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %31 = tt.splat %28 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %32 = arith.addi %29, %17 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %33 = arith.addi %30, %19 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %34 = arith.addi %31, %21 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %35 = tt.expand_dims %25 {axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<128x1xi32, #blocked>
    %36 = tt.expand_dims %26 {axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi32, #blocked1>
    %37 = tt.expand_dims %27 {axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi32, #blocked1>
    %38 = tt.splat %arg6 : (i32) -> tensor<128x1xi32, #blocked>
    %39 = arith.muli %35, %38 : tensor<128x1xi32, #blocked>
    %40 = tt.splat %arg0 : (!tt.ptr<f16>) -> tensor<128x1x!tt.ptr<f16>, #blocked>
    %41 = tt.addptr %40, %39 : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi32, #blocked>
    %42 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %43 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %44 = tt.expand_dims %42 {axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x32xi32, #blocked>
    %45 = tt.expand_dims %43 {axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x32xi32, #blocked>
    %46 = tt.broadcast %41 : (tensor<128x1x!tt.ptr<f16>, #blocked>) -> tensor<128x32x!tt.ptr<f16>, #blocked>
    %47 = tt.broadcast %44 : (tensor<1x32xi32, #blocked>) -> tensor<128x32xi32, #blocked>
    %48 = tt.addptr %46, %47 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
    %49 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %50 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %51 = tt.expand_dims %49 {axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<32x1xi32, #blocked1>
    %52 = tt.expand_dims %50 {axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<32x1xi32, #blocked1>
    %53 = tt.splat %arg7 : (i32) -> tensor<32x1xi32, #blocked1>
    %54 = arith.muli %51, %53 : tensor<32x1xi32, #blocked1>
    %55 = tt.splat %arg1 : (!tt.ptr<f16>) -> tensor<32x1x!tt.ptr<f16>, #blocked1>
    %56 = tt.addptr %55, %54 : tensor<32x1x!tt.ptr<f16>, #blocked1>, tensor<32x1xi32, #blocked1>
    %57 = tt.expand_dims %32 {axis = 0 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x128xi32, #blocked1>
    %58 = tt.expand_dims %33 {axis = 0 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x128xi32, #blocked1>
    %59 = tt.expand_dims %34 {axis = 0 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x128xi32, #blocked1>
    %60 = tt.broadcast %56 : (tensor<32x1x!tt.ptr<f16>, #blocked1>) -> tensor<32x128x!tt.ptr<f16>, #blocked1>
    %61 = tt.broadcast %57 : (tensor<1x128xi32, #blocked1>) -> tensor<32x128xi32, #blocked1>
    %62 = tt.addptr %60, %61 : tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<32x128xi32, #blocked1>
    %63 = arith.addi %arg5, %c31_i32 : i32
    %64 = arith.divsi %63, %c32_i32 : i32
    %65 = arith.truncf %cst_1 : tensor<128x32xf32, #blocked> to tensor<128x32xf16, #blocked>
    %66 = arith.truncf %cst_2 : tensor<32x128xf32, #blocked1> to tensor<32x128xf16, #blocked1>
    %67 = arith.muli %arg7, %c32_i32 : i32
    %68 = tt.splat %67 : (i32) -> tensor<32x128xi32, #blocked1>
    %69:3 = scf.for %arg9 = %c0_i32 to %64 step %c1_i32 iter_args(%arg10 = %cst, %arg11 = %48, %arg12 = %62) -> (tensor<128x128xf32, #triton_gpu.mfma<{nonKDim = 32, warpsPerCTA = [2, 2], isTransposed = false}>>, tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>)  : i32 {
      %86 = arith.muli %arg9, %c32_i32 : i32
      %87 = arith.subi %arg5, %86 : i32
      %88 = tt.splat %87 : (i32) -> tensor<1x32xi32, #blocked>
      %89 = "triton_gpu.cmpi"(%45, %88) <{predicate = 2 : i64}> : (tensor<1x32xi32, #blocked>, tensor<1x32xi32, #blocked>) -> tensor<1x32xi1, #blocked>
      %90 = tt.broadcast %89 : (tensor<1x32xi1, #blocked>) -> tensor<128x32xi1, #blocked>
      %91 = tt.load %arg11, %90, %65 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked>
      %92 = tt.splat %87 : (i32) -> tensor<32x1xi32, #blocked1>
      %93 = "triton_gpu.cmpi"(%52, %92) <{predicate = 2 : i64}> : (tensor<32x1xi32, #blocked1>, tensor<32x1xi32, #blocked1>) -> tensor<32x1xi1, #blocked1>
      %94 = tt.broadcast %93 : (tensor<32x1xi1, #blocked1>) -> tensor<32x128xi1, #blocked1>
      %95 = tt.load %arg12, %94, %66 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked1>
      %96 = triton_gpu.convert_layout %91 : (tensor<128x32xf16, #blocked>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #triton_gpu.mfma<{nonKDim = 32, warpsPerCTA = [2, 2], isTransposed = false}>}>>
      %97 = triton_gpu.convert_layout %95 : (tensor<32x128xf16, #blocked1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #triton_gpu.mfma<{nonKDim = 32, warpsPerCTA = [2, 2], isTransposed = false}>}>>
      %98 = tt.dot %96, %97, %arg10 {allowTF32 = true} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #triton_gpu.mfma<{nonKDim = 32, warpsPerCTA = [2, 2], isTransposed = false}>}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #triton_gpu.mfma<{nonKDim = 32, warpsPerCTA = [2, 2], isTransposed = false}>}>> -> tensor<128x128xf32, #triton_gpu.mfma<{nonKDim = 32, warpsPerCTA = [2, 2], isTransposed = false}>>
      %99 = tt.addptr %arg11, %cst_0 : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
      %100 = tt.addptr %arg12, %68 : tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<32x128xi32, #blocked1>
      scf.yield %98, %99, %100 : tensor<128x128xf32, #triton_gpu.mfma<{nonKDim = 32, warpsPerCTA = [2, 2], isTransposed = false}>>, tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>
    }
    %70 = arith.truncf %69#0 : tensor<128x128xf32, #triton_gpu.mfma<{nonKDim = 32, warpsPerCTA = [2, 2], isTransposed = false}>> to tensor<128x128xf16, #triton_gpu.mfma<{nonKDim = 32, warpsPerCTA = [2, 2], isTransposed = false}>>
    %71 = tt.splat %arg8 : (i32) -> tensor<128x1xi32, #blocked1>
    %72 = arith.muli %71, %36 : tensor<128x1xi32, #blocked1>
    %73 = tt.splat %arg2 : (!tt.ptr<f16>) -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %74 = tt.addptr %73, %72 : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
    %75 = tt.broadcast %74 : (tensor<128x1x!tt.ptr<f16>, #blocked1>) -> tensor<128x128x!tt.ptr<f16>, #blocked1>
    %76 = tt.broadcast %58 : (tensor<1x128xi32, #blocked1>) -> tensor<128x128xi32, #blocked1>
    %77 = tt.addptr %75, %76 : tensor<128x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xi32, #blocked1>
    %78 = tt.splat %arg3 : (i32) -> tensor<128x1xi32, #blocked1>
    %79 = "triton_gpu.cmpi"(%37, %78) <{predicate = 2 : i64}> : (tensor<128x1xi32, #blocked1>, tensor<128x1xi32, #blocked1>) -> tensor<128x1xi1, #blocked1>
    %80 = tt.splat %arg4 : (i32) -> tensor<1x128xi32, #blocked1>
    %81 = "triton_gpu.cmpi"(%59, %80) <{predicate = 2 : i64}> : (tensor<1x128xi32, #blocked1>, tensor<1x128xi32, #blocked1>) -> tensor<1x128xi1, #blocked1>
    %82 = tt.broadcast %79 : (tensor<128x1xi1, #blocked1>) -> tensor<128x128xi1, #blocked1>
    %83 = tt.broadcast %81 : (tensor<1x128xi1, #blocked1>) -> tensor<128x128xi1, #blocked1>
    %84 = arith.andi %82, %83 : tensor<128x128xi1, #blocked1>
    %85 = triton_gpu.convert_layout %70 : (tensor<128x128xf16, #triton_gpu.mfma<{nonKDim = 32, warpsPerCTA = [2, 2], isTransposed = false}>>) -> tensor<128x128xf16, #blocked1>
    tt.store %77, %85, %84 {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf16, #blocked1>
    tt.return
  }
}


