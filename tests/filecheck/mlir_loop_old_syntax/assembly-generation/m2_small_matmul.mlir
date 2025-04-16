// RUN: mlir-loop --old-syntax %s --no-alias --arch aarch64 --cpu apple-m2 --print-assembly --hide-jumps 2>&1 | filecheck %s

func.func @myfun(
  %A: memref<4x4xf32>,
  %B: memref<4x4xf32>,
  %C: memref<4x4xf32>
) {
  linalg.matmul
    {
      loop.dims = ["i","j","k"],
      loop.interchange = ["i","k","j"],
      loop.vectorize = ["j"]
    }
    ins(%A, %B : memref<4x4xf32>, memref<4x4xf32>)
    outs(%C : memref<4x4xf32>)
  return
}
// CHECK:       <myfun>:
// CHECK-NEXT:  	ldp	s2, s3, [x0]
// CHECK-NEXT:  	ldp	q1, q0, [x1]
// CHECK-NEXT:  	ldp	q4, q5, [x2]
// CHECK-NEXT:  	fmla	v4.4s, v1.4s, v2.s[0]
// CHECK-NEXT:  	fmla	v4.4s, v0.4s, v3.s[0]
// CHECK-NEXT:  	ldp	s3, s6, [x0, #8]
// CHECK-NEXT:  	ldp	q7, q2, [x1, #32]
// CHECK-NEXT:  	fmla	v4.4s, v7.4s, v3.s[0]
// CHECK-NEXT:  	fmla	v4.4s, v2.4s, v6.s[0]
// CHECK-NEXT:  	ldp	s3, s6, [x0, #16]
// CHECK-NEXT:  	fmla	v5.4s, v1.4s, v3.s[0]
// CHECK-NEXT:  	fmla	v5.4s, v0.4s, v6.s[0]
// CHECK-NEXT:  	ldp	s3, s6, [x0, #24]
// CHECK-NEXT:  	fmla	v5.4s, v7.4s, v3.s[0]
// CHECK-NEXT:  	fmla	v5.4s, v2.4s, v6.s[0]
// CHECK-NEXT:  	stp	q4, q5, [x2]
// CHECK-NEXT:  	ldp	s3, s4, [x0, #32]
// CHECK-NEXT:  	ldp	q5, q6, [x2, #32]
// CHECK-NEXT:  	fmla	v5.4s, v1.4s, v3.s[0]
// CHECK-NEXT:  	fmla	v5.4s, v0.4s, v4.s[0]
// CHECK-NEXT:  	ldp	s3, s4, [x0, #48]
// CHECK-NEXT:  	fmla	v6.4s, v1.4s, v3.s[0]
// CHECK-NEXT:  	ldp	s1, s3, [x0, #40]
// CHECK-NEXT:  	fmla	v5.4s, v7.4s, v1.s[0]
// CHECK-NEXT:  	fmla	v5.4s, v2.4s, v3.s[0]
// CHECK-NEXT:  	fmla	v6.4s, v0.4s, v4.s[0]
// CHECK-NEXT:  	ldp	s0, s1, [x0, #56]
// CHECK-NEXT:  	fmla	v6.4s, v7.4s, v0.s[0]
// CHECK-NEXT:  	fmla	v6.4s, v2.4s, v1.s[0]
// CHECK-NEXT:  	stp	q5, q6, [x2, #32]
// CHECK-NEXT:  	ret
