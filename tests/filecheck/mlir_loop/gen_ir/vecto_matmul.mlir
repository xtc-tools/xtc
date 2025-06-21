// RUN: mlir-loop --no-alias --print-transformed-ir %s 2>&1 | grep "vector\." | filecheck %s

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

// CHECK:      %{{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [false, true, false], permutation_map = #map} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x64x1xf32>
// CHECK-NEXT: %{{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [true, false, false], permutation_map = #map1} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64x1xf32>
// CHECK-NEXT: %{{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT: %{{.*}} = vector.multi_reduction <add>, %{{.*}}, %{{.*}} [2] : vector<1x64x1xf32> to vector<1x64xf32>
// CHECK-NEXT: vector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT: %{{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [false, true, false], permutation_map = #map} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x64x1xf32>
// CHECK-NEXT: %{{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [true, false, false], permutation_map = #map1} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64x1xf32>
// CHECK-NEXT: %{{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT: %{{.*}} = vector.multi_reduction <add>, %{{.*}}, %{{.*}} [2] : vector<1x64x1xf32> to vector<1x64xf32>
// CHECK-NEXT: vector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT: %{{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [false, true, false], permutation_map = #map} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x64x1xf32>
// CHECK-NEXT: %{{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [true, false, false], permutation_map = #map1} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64x1xf32>
// CHECK-NEXT: %{{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT: %{{.*}} = vector.multi_reduction <add>, %{{.*}}, %{{.*}} [2] : vector<1x64x1xf32> to vector<1x64xf32>
// CHECK-NEXT: vector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT: %{{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [false, true, false], permutation_map = #map} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x64x1xf32>
// CHECK-NEXT: %{{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [true, false, false], permutation_map = #map1} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64x1xf32>
// CHECK-NEXT: %{{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT: %{{.*}} = vector.multi_reduction <add>, %{{.*}}, %{{.*}} [2] : vector<1x64x1xf32> to vector<1x64xf32>
// CHECK-NEXT: vector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT: %{{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [false, true, false], permutation_map = #map} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x64x1xf32>
// CHECK-NEXT: %{{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [true, false, false], permutation_map = #map1} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64x1xf32>
// CHECK-NEXT: %{{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT: %{{.*}} = vector.multi_reduction <add>, %{{.*}}, %{{.*}} [2] : vector<1x64x1xf32> to vector<1x64xf32>
// CHECK-NEXT: vector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT: %{{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [false, true, false], permutation_map = #map} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x64x1xf32>
// CHECK-NEXT: %{{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [true, false, false], permutation_map = #map1} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64x1xf32>
// CHECK-NEXT: %{{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT: %{{.*}} = vector.multi_reduction <add>, %{{.*}}, %{{.*}} [2] : vector<1x64x1xf32> to vector<1x64xf32>
// CHECK-NEXT: vector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT: %{{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [false, true, false], permutation_map = #map} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x64x1xf32>
// CHECK-NEXT: %{{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [true, false, false], permutation_map = #map1} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64x1xf32>
// CHECK-NEXT: %{{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT: %{{.*}} = vector.multi_reduction <add>, %{{.*}}, %{{.*}} [2] : vector<1x64x1xf32> to vector<1x64xf32>
// CHECK-NEXT: vector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT: %{{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [false, true, false], permutation_map = #map} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x64x1xf32>
// CHECK-NEXT: %{{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [true, false, false], permutation_map = #map1} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64x1xf32>
// CHECK-NEXT: %{{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT: %{{.*}} = vector.multi_reduction <add>, %{{.*}}, %{{.*}} [2] : vector<1x64x1xf32> to vector<1x64xf32>
// CHECK-NEXT: vector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
