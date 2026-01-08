// RUN: not mlir-loop --no-alias --print-source-ir --print-transformed-ir %s 2>&1 | filecheck %s
func.func @matmul(%A: memref<256x512xf64>, %B: memref<512x256xf64>, %C: memref<256x256xf64>){
  linalg.matmul {
    loop.dims = ["i", "j"],
    loop.schedule = {
      "i[0:5]" = { "j" = {"unroll"} },
      "i[5:]" = { "j" }
    }
  }
  ins(%A, %B : memref<256x512xf64>, memref<512x256xf64>)
  outs(%C: memref<256x256xf64>)
  return
}
// CHECK:      j cannot be implicitly fully unrolled if its
// CHECK-NEXT: size is unknown (needs an unroll factor)
