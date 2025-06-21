// RUN: mlir-loop --no-alias --print-lowered-ir %s 2>&1 | grep omp | filecheck %s

func.func @myfun(
  %A: memref<256x512xf32>,
  %B: memref<512x256xf32>,
  %C: memref<256x256xf32>
) {
  linalg.matmul
    {
      loop.dims = ["I","J","K"],
      loop.schedule = {
        "I" = {"parallelize"},
          "J",
            "K",
              "I#1" = {"unroll"},
                "K#8"= {"unroll"},
                  "J#64" = {"vectorize"}
      }
    }
    ins(%A, %B : memref<256x512xf32>, memref<512x256xf32>)
    outs(%C : memref<256x256xf32>)
  return
}
// CHECK:      omp.parallel {
// CHECK-NEXT: omp.wsloop {
// CHECK-NEXT: omp.loop_nest (%{{.*}}) : i64 = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) {
// CHECK-NEXT: omp.yield
// CHECK-NEXT: omp.terminator
// CHECK-NEXT: omp.terminator
