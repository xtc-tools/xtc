// RUN: mlir-loop --no-alias --print-transformed-ir %s 2>&1 | grep fma | filecheck %s

func.func @myfun(
  %A: memref<256x512xf32>,
  %B: memref<512x256xf32>,
  %C: memref<256x256xf32>
) {
  linalg.matmul
    {
      loop.dims = ["I","J","K"],
      loop.schedule = {
        "I",
          "J",
            "K",
              "I#1" = {"unroll"},
                "K#8"= {"unroll"},
                  "J#64" = {"unroll","vectorize"}
      }
    }
    ins(%A, %B : memref<256x512xf32>, memref<512x256xf32>)
    outs(%C : memref<256x256xf32>)
  return
}
// CHECK:                 %8 = vector.fma %6, %4, %7 : vector<64xf32>
// CHECK-NEXT:            %17 = vector.fma %15, %13, %16 : vector<64xf32>
// CHECK-NEXT:            %26 = vector.fma %24, %22, %25 : vector<64xf32>
// CHECK-NEXT:            %35 = vector.fma %33, %31, %34 : vector<64xf32>
// CHECK-NEXT:            %44 = vector.fma %42, %40, %43 : vector<64xf32>
// CHECK-NEXT:            %53 = vector.fma %51, %49, %52 : vector<64xf32>
// CHECK-NEXT:            %62 = vector.fma %60, %58, %61 : vector<64xf32>
// CHECK-NEXT:            %71 = vector.fma %69, %67, %70 : vector<64xf32>
