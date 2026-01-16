// RUN: mlir-loop --no-alias --evaluate %s 2>&1 | filecheck %s
// UNSUPPORTED: mlir-target=nvgpu

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
    ins(%A, %B: memref<256x512xf32>, memref<512x256xf32>)
    outs(%C: memref<256x256xf32>)
  return
}

// CHECK: [[FLOAT:.*[0-9]+\.[0-9]+.*]]
