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

// CHECK:                 %1 = vector.transfer_read %subview_6[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %2 = vector.transfer_read %subview_7[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %3 = vector.transfer_read %subview_3[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %4 = vector.extract %2[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %5 = vector.extract %1[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %6 = vector.broadcast %5 : f32 to vector<64xf32>
// CHECK-NEXT:            %7 = vector.extract %3[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %8 = vector.fma %6, %4, %7 : vector<64xf32>
// CHECK-NEXT:            %9 = vector.insert %8, %cst [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %9, %subview_3[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %10 = vector.transfer_read %subview_8[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %11 = vector.transfer_read %subview_9[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %12 = vector.transfer_read %subview_3[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %13 = vector.extract %11[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %14 = vector.extract %10[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %15 = vector.broadcast %14 : f32 to vector<64xf32>
// CHECK-NEXT:            %16 = vector.extract %12[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %17 = vector.fma %15, %13, %16 : vector<64xf32>
// CHECK-NEXT:            %18 = vector.insert %17, %cst [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %18, %subview_3[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %19 = vector.transfer_read %subview_10[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %20 = vector.transfer_read %subview_11[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %21 = vector.transfer_read %subview_3[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %22 = vector.extract %20[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %23 = vector.extract %19[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %24 = vector.broadcast %23 : f32 to vector<64xf32>
// CHECK-NEXT:            %25 = vector.extract %21[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %26 = vector.fma %24, %22, %25 : vector<64xf32>
// CHECK-NEXT:            %27 = vector.insert %26, %cst [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %27, %subview_3[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %28 = vector.transfer_read %subview_12[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %29 = vector.transfer_read %subview_13[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %30 = vector.transfer_read %subview_3[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %31 = vector.extract %29[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %32 = vector.extract %28[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %33 = vector.broadcast %32 : f32 to vector<64xf32>
// CHECK-NEXT:            %34 = vector.extract %30[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %35 = vector.fma %33, %31, %34 : vector<64xf32>
// CHECK-NEXT:            %36 = vector.insert %35, %cst [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %36, %subview_3[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %37 = vector.transfer_read %subview_14[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %38 = vector.transfer_read %subview_15[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %39 = vector.transfer_read %subview_3[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %40 = vector.extract %38[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %41 = vector.extract %37[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %42 = vector.broadcast %41 : f32 to vector<64xf32>
// CHECK-NEXT:            %43 = vector.extract %39[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %44 = vector.fma %42, %40, %43 : vector<64xf32>
// CHECK-NEXT:            %45 = vector.insert %44, %cst [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %45, %subview_3[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %46 = vector.transfer_read %subview_16[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %47 = vector.transfer_read %subview_17[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %48 = vector.transfer_read %subview_3[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %49 = vector.extract %47[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %50 = vector.extract %46[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %51 = vector.broadcast %50 : f32 to vector<64xf32>
// CHECK-NEXT:            %52 = vector.extract %48[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %53 = vector.fma %51, %49, %52 : vector<64xf32>
// CHECK-NEXT:            %54 = vector.insert %53, %cst [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %54, %subview_3[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %55 = vector.transfer_read %subview_18[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %56 = vector.transfer_read %subview_19[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %57 = vector.transfer_read %subview_3[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %58 = vector.extract %56[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %59 = vector.extract %55[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %60 = vector.broadcast %59 : f32 to vector<64xf32>
// CHECK-NEXT:            %61 = vector.extract %57[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %62 = vector.fma %60, %58, %61 : vector<64xf32>
// CHECK-NEXT:            %63 = vector.insert %62, %cst [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %63, %subview_3[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
// CHECK-NEXT:            %64 = vector.transfer_read %subview_20[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x1xf32>
// CHECK-NEXT:            %65 = vector.transfer_read %subview_21[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %66 = vector.transfer_read %subview_3[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>>, vector<1x64xf32>
// CHECK-NEXT:            %67 = vector.extract %65[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %68 = vector.extract %64[0, 0] : f32 from vector<1x1xf32>
// CHECK-NEXT:            %69 = vector.broadcast %68 : f32 to vector<64xf32>
// CHECK-NEXT:            %70 = vector.extract %66[0] : vector<64xf32> from vector<1x64xf32>
// CHECK-NEXT:            %71 = vector.fma %69, %67, %70 : vector<64xf32>
// CHECK-NEXT:            %72 = vector.insert %71, %cst [0] : vector<64xf32> into vector<1x64xf32>
// CHECK-NEXT:            vector.transfer_write %72, %subview_3[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>>
