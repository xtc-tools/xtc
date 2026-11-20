// RUN: not mlir-loop --no-alias --print-source-ir %s 2>&1 | filecheck %s
func.func @matmul(%A: memref<256x512xf64>, %B: memref<512x256xf64>, %C: memref<256x256xf64>){
  linalg.matmul {
    loop.dims = ["i", "j", "k"],
    loop.schedule = {
      "i",
        "j" = {"vectorize"},
          "k[:128]" = {
            "i" = {"vectorize"}
          },
          "k[:]" = {
            "i"
          }
    }
  }
  ins(%A, %B: memref<256x512xf64>, memref<512x256xf64>)
  outs(%C: memref<256x256xf64>)
  return
}
// CHECK:  Vector annotations are only allowed in leaf nodes; __node0__ declares a partition.
