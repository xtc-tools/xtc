// RUN: mlir-loop --no-alias --arch aarch64 --cpu cortex-a72 --print-assembly --hide-jumps %s 2>&1 | filecheck %s
// UNSUPPORTED: mlir-target=c
// Missing a C cross compiler as dependancy.

func.func @myfun(
  %A: memref<4x4xf32>,
  %B: memref<4x4xf32>,
  %C: memref<4x4xf32>
) {
  linalg.matmul
    {
      loop.dims = ["I","J","K"],
      loop.schedule = {
        "I",
          "K",
            "J" = {"vectorize"}
      }
    }
    ins(%A, %B: memref<4x4xf32>, memref<4x4xf32>)
    outs(%C: memref<4x4xf32>)
  return
}
// CHECK:       <myfun>:
// CHECK-NEXT:  	ldp	s4, s3, [x0]
// CHECK-NEXT:  	ldp	q5, q1, [x1]
// CHECK-NEXT:  	ldp	q0, q2, [x2]
// CHECK-NEXT:  	ldp	s6, s7, [x0, #16]
// CHECK-NEXT:  	ldp	s16, s17, [x0, #32]
// CHECK-NEXT:  	fmla	v0.4s, v5.4s, v4.s[0]
// CHECK-NEXT:  	ldp	q4, q18, [x2, #32]
// CHECK-NEXT:  	fmla	v2.4s, v5.4s, v6.s[0]
// CHECK-NEXT:  	fmla	v4.4s, v5.4s, v16.s[0]
// CHECK-NEXT:  	ldp	s6, s16, [x0, #48]
// CHECK-NEXT:  	fmla	v18.4s, v5.4s, v6.s[0]
// CHECK-NEXT:  	fmla	v0.4s, v1.4s, v3.s[0]
// CHECK-NEXT:  	fmla	v2.4s, v1.4s, v7.s[0]
// CHECK-NEXT:  	ldp	s3, s5, [x0, #8]
// CHECK-NEXT:  	fmla	v4.4s, v1.4s, v17.s[0]
// CHECK-NEXT:  	fmla	v18.4s, v1.4s, v16.s[0]
// CHECK-NEXT:  	ldp	q1, q6, [x1, #32]
// CHECK-NEXT:  	fmla	v0.4s, v1.4s, v3.s[0]
// CHECK-NEXT:  	ldp	s3, s7, [x0, #24]
// CHECK-NEXT:  	fmla	v2.4s, v1.4s, v3.s[0]
// CHECK-NEXT:  	fmla	v0.4s, v6.4s, v5.s[0]
// CHECK-NEXT:  	fmla	v2.4s, v6.4s, v7.s[0]
// CHECK-NEXT:  	ldp	s3, s5, [x0, #40]
// CHECK-NEXT:  	stp	q0, q2, [x2]
// CHECK-NEXT:  	fmla	v4.4s, v1.4s, v3.s[0]
// CHECK-NEXT:  	ldp	s0, s2, [x0, #56]
// CHECK-NEXT:  	fmla	v18.4s, v1.4s, v0.s[0]
// CHECK-NEXT:  	fmla	v4.4s, v6.4s, v5.s[0]
// CHECK-NEXT:  	fmla	v18.4s, v6.4s, v2.s[0]
// CHECK-NEXT:  	stp	q4, q18, [x2, #32]
// CHECK-NEXT:  	ret
