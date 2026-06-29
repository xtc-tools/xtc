// RUN: mlir-loop --no-alias --arch x86-64 --cpu skylake --print-assembly --hide-jumps %s 2>&1 | grep -v '\(nop\|ret\)' | filecheck %s
// REQUIRES: mlir-target=llvmir
// Assembly output will differ a bit when using C.

func.func @myfun(
  %I: memref<1x30x66x64xf32>,
  %K: memref<3x3x64x128xf32>,
  %O: memref<1x28x64x128xf32>
) {
  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>,
      affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>,
      affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
    ],
    iterator_types = [
      "parallel",
      "parallel",
      "parallel",
      "parallel",
      "reduction",
      "reduction",
      "reduction"
    ]
  }
    ins(%I, %K : memref<1x30x66x64xf32>, memref<3x3x64x128xf32>)
    outs(%O : memref<1x28x64x128xf32>)
      attrs = {
        loop.dims = ["n","h","w","f","r","s","c"],
        loop.schedule = {
          "n",
            "h",
              "f",
                "r",
                  "s",
                    "c",
                      "w",
                        "f#1" = {"unroll"},
                          "c#8" = {"unroll"},
                            "w#64" = {"vectorize"}
        }
      }
    {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    }
  return
}
// CHECK: Disassembly of section .text:
