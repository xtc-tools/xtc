// RUN: mlir-loop --no-alias --arch aarch64 --cpu cortex-a72 --print-assembly --hide-jumps %s 2>&1 | filecheck %s

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
    ins(%A, %B : memref<4x4xf32>, memref<4x4xf32>)
    outs(%C : memref<4x4xf32>)
  return
}
// CHECK:       <myfun>:
// CHECK-NEXT:  	sub	sp, sp, #0x50
// CHECK-NEXT:  	stp	d15, d14, [sp, #16]
// CHECK-NEXT:  	stp	d13, d12, [sp, #32]
// CHECK-NEXT:  	stp	d11, d10, [sp, #48]
// CHECK-NEXT:  	stp	d9, d8, [sp, #64]
// CHECK-NEXT:  	ldp	s0, s20, [x1]
// CHECK-NEXT:  	ldp	s16, s22, [x0]
// CHECK-NEXT:  	ldp	s24, s23, [x0, #8]
// CHECK-NEXT:  	ldp	s2, s1, [x1]
// CHECK-NEXT:  	stp	s1, s2, [sp, #8]
// CHECK-NEXT:  	ldp	s26, s27, [x1, #8]
// CHECK-NEXT:  	ldp	s1, s4, [x1, #8]
// CHECK-NEXT:  	str	s1, [sp, #4]
// CHECK-NEXT:  	ldp	s28, s29, [x1, #16]
// CHECK-NEXT:  	ldp	s19, s21, [x1, #16]
// CHECK-NEXT:  	ldp	s30, s8, [x1, #24]
// CHECK-NEXT:  	ldp	s18, s17, [x1, #24]
// CHECK-NEXT:  	ldp	s31, s9, [x1, #32]
// CHECK-NEXT:  	ldp	s5, s6, [x1, #32]
// CHECK-NEXT:  	ldp	s10, s11, [x1, #40]
// CHECK-NEXT:  	ldr	s7, [x1, #40]
// CHECK-NEXT:  	fmul	s25, s16, s0
// CHECK-NEXT:  	fmul	s12, s16, s20
// CHECK-NEXT:  	fmul	s13, s16, s26
// CHECK-NEXT:  	fmul	s14, s22, s28
// CHECK-NEXT:  	fmul	s15, s22, s29
// CHECK-NEXT:  	mov	v25.s[1], v12.s[0]
// CHECK-NEXT:  	fmul	s12, s24, s31
// CHECK-NEXT:  	mov	v14.s[1], v15.s[0]
// CHECK-NEXT:  	fmul	s15, s24, s9
// CHECK-NEXT:  	mov	v12.s[1], v15.s[0]
// CHECK-NEXT:  	fmul	s16, s16, s27
// CHECK-NEXT:  	mov	v25.s[2], v13.s[0]
// CHECK-NEXT:  	fmul	s13, s22, s30
// CHECK-NEXT:  	mov	v14.s[2], v13.s[0]
// CHECK-NEXT:  	fmul	s13, s24, s10
// CHECK-NEXT:  	mov	v12.s[2], v13.s[0]
// CHECK-NEXT:  	fmul	s22, s22, s8
// CHECK-NEXT:  	mov	v25.s[3], v16.s[0]
// CHECK-NEXT:  	fmul	s16, s24, s11
// CHECK-NEXT:  	mov	v14.s[3], v22.s[0]
// CHECK-NEXT:  	mov	v12.s[3], v16.s[0]
// CHECK-NEXT:  	ldp	q16, q13, [x2]
// CHECK-NEXT:  	fadd	v22.4s, v16.4s, v25.4s
// CHECK-NEXT:  	fadd	v24.4s, v14.4s, v12.4s
// CHECK-NEXT:  	ldp	s16, s12, [x1, #44]
// CHECK-NEXT:  	fadd	v14.4s, v22.4s, v24.4s
// CHECK-NEXT:  	fmul	s15, s23, s12
// CHECK-NEXT:  	ldp	s22, s3, [x1, #48]
// CHECK-NEXT:  	fmul	s24, s23, s3
// CHECK-NEXT:  	mov	v15.s[1], v24.s[0]
// CHECK-NEXT:  	ldp	s25, s2, [x1, #52]
// CHECK-NEXT:  	fmul	s24, s23, s2
// CHECK-NEXT:  	mov	v15.s[2], v24.s[0]
// CHECK-NEXT:  	ldp	s24, s1, [x1, #56]
// CHECK-NEXT:  	fmul	s23, s23, s1
// CHECK-NEXT:  	mov	v15.s[3], v23.s[0]
// CHECK-NEXT:  	fadd	v23.4s, v14.4s, v15.4s
// CHECK-NEXT:  	ldp	s14, s15, [x0, #16]
// CHECK-NEXT:  	fmul	s0, s14, s0
// CHECK-NEXT:  	fmul	s20, s14, s20
// CHECK-NEXT:  	mov	v0.s[1], v20.s[0]
// CHECK-NEXT:  	fmul	s20, s14, s26
// CHECK-NEXT:  	mov	v0.s[2], v20.s[0]
// CHECK-NEXT:  	fmul	s20, s14, s27
// CHECK-NEXT:  	mov	v0.s[3], v20.s[0]
// CHECK-NEXT:  	fadd	v0.4s, v13.4s, v0.4s
// CHECK-NEXT:  	fmul	s20, s15, s28
// CHECK-NEXT:  	fmul	s26, s15, s29
// CHECK-NEXT:  	mov	v20.s[1], v26.s[0]
// CHECK-NEXT:  	fmul	s26, s15, s30
// CHECK-NEXT:  	mov	v20.s[2], v26.s[0]
// CHECK-NEXT:  	fmul	s26, s15, s8
// CHECK-NEXT:  	mov	v20.s[3], v26.s[0]
// CHECK-NEXT:  	ldp	s26, s27, [x0, #24]
// CHECK-NEXT:  	fmul	s28, s26, s31
// CHECK-NEXT:  	fmul	s29, s26, s9
// CHECK-NEXT:  	mov	v28.s[1], v29.s[0]
// CHECK-NEXT:  	fmul	s29, s26, s10
// CHECK-NEXT:  	mov	v28.s[2], v29.s[0]
// CHECK-NEXT:  	fmul	s26, s26, s11
// CHECK-NEXT:  	mov	v28.s[3], v26.s[0]
// CHECK-NEXT:  	fadd	v20.4s, v20.4s, v28.4s
// CHECK-NEXT:  	fadd	v0.4s, v0.4s, v20.4s
// CHECK-NEXT:  	fmul	s20, s27, s12
// CHECK-NEXT:  	fmul	s3, s27, s3
// CHECK-NEXT:  	mov	v20.s[1], v3.s[0]
// CHECK-NEXT:  	fmul	s2, s27, s2
// CHECK-NEXT:  	mov	v20.s[2], v2.s[0]
// CHECK-NEXT:  	fmul	s1, s27, s1
// CHECK-NEXT:  	mov	v20.s[3], v1.s[0]
// CHECK-NEXT:  	fadd	v0.4s, v0.4s, v20.4s
// CHECK-NEXT:  	stp	q23, q0, [x2]
// CHECK-NEXT:  	ldp	s0, s1, [x0, #32]
// CHECK-NEXT:  	ldp	s28, s27, [sp, #8]
// CHECK-NEXT:  	fmul	s2, s0, s27
// CHECK-NEXT:  	fmul	s3, s0, s28
// CHECK-NEXT:  	mov	v2.s[1], v3.s[0]
// CHECK-NEXT:  	ldr	s29, [sp, #4]
// CHECK-NEXT:  	fmul	s3, s0, s29
// CHECK-NEXT:  	mov	v2.s[2], v3.s[0]
// CHECK-NEXT:  	fmul	s0, s0, s4
// CHECK-NEXT:  	mov	v2.s[3], v0.s[0]
// CHECK-NEXT:  	ldp	q0, q3, [x2, #32]
// CHECK-NEXT:  	fadd	v0.4s, v0.4s, v2.4s
// CHECK-NEXT:  	fmul	s2, s1, s19
// CHECK-NEXT:  	fmul	s20, s1, s21
// CHECK-NEXT:  	mov	v2.s[1], v20.s[0]
// CHECK-NEXT:  	fmul	s20, s1, s18
// CHECK-NEXT:  	mov	v2.s[2], v20.s[0]
// CHECK-NEXT:  	fmul	s1, s1, s17
// CHECK-NEXT:  	ldp	s20, s23, [x0, #40]
// CHECK-NEXT:  	fmul	s26, s20, s5
// CHECK-NEXT:  	mov	v2.s[3], v1.s[0]
// CHECK-NEXT:  	fmul	s1, s20, s6
// CHECK-NEXT:  	mov	v26.s[1], v1.s[0]
// CHECK-NEXT:  	fmul	s1, s20, s7
// CHECK-NEXT:  	mov	v26.s[2], v1.s[0]
// CHECK-NEXT:  	fmul	s1, s20, s16
// CHECK-NEXT:  	mov	v26.s[3], v1.s[0]
// CHECK-NEXT:  	fadd	v1.4s, v2.4s, v26.4s
// CHECK-NEXT:  	fadd	v0.4s, v0.4s, v1.4s
// CHECK-NEXT:  	fmul	s1, s23, s22
// CHECK-NEXT:  	fmul	s2, s23, s25
// CHECK-NEXT:  	fmul	s26, s23, s24
// CHECK-NEXT:  	mov	v1.s[1], v2.s[0]
// CHECK-NEXT:  	ldr	s20, [x1, #60]
// CHECK-NEXT:  	mov	v1.s[2], v26.s[0]
// CHECK-NEXT:  	fmul	s2, s23, s20
// CHECK-NEXT:  	mov	v1.s[3], v2.s[0]
// CHECK-NEXT:  	fadd	v0.4s, v0.4s, v1.4s
// CHECK-NEXT:  	ldp	s1, s2, [x0, #48]
// CHECK-NEXT:  	fmul	s23, s1, s27
// CHECK-NEXT:  	fmul	s26, s1, s28
// CHECK-NEXT:  	mov	v23.s[1], v26.s[0]
// CHECK-NEXT:  	fmul	s26, s1, s29
// CHECK-NEXT:  	mov	v23.s[2], v26.s[0]
// CHECK-NEXT:  	fmul	s1, s1, s4
// CHECK-NEXT:  	mov	v23.s[3], v1.s[0]
// CHECK-NEXT:  	fadd	v1.4s, v3.4s, v23.4s
// CHECK-NEXT:  	fmul	s3, s2, s19
// CHECK-NEXT:  	fmul	s4, s2, s21
// CHECK-NEXT:  	mov	v3.s[1], v4.s[0]
// CHECK-NEXT:  	fmul	s4, s2, s18
// CHECK-NEXT:  	mov	v3.s[2], v4.s[0]
// CHECK-NEXT:  	fmul	s2, s2, s17
// CHECK-NEXT:  	ldp	s4, s17, [x0, #56]
// CHECK-NEXT:  	fmul	s5, s4, s5
// CHECK-NEXT:  	mov	v3.s[3], v2.s[0]
// CHECK-NEXT:  	fmul	s2, s4, s6
// CHECK-NEXT:  	mov	v5.s[1], v2.s[0]
// CHECK-NEXT:  	fmul	s2, s4, s7
// CHECK-NEXT:  	mov	v5.s[2], v2.s[0]
// CHECK-NEXT:  	fmul	s2, s4, s16
// CHECK-NEXT:  	mov	v5.s[3], v2.s[0]
// CHECK-NEXT:  	fadd	v2.4s, v3.4s, v5.4s
// CHECK-NEXT:  	fadd	v1.4s, v1.4s, v2.4s
// CHECK-NEXT:  	fmul	s2, s17, s22
// CHECK-NEXT:  	fmul	s3, s17, s25
// CHECK-NEXT:  	mov	v2.s[1], v3.s[0]
// CHECK-NEXT:  	fmul	s3, s17, s24
// CHECK-NEXT:  	mov	v2.s[2], v3.s[0]
// CHECK-NEXT:  	fmul	s3, s17, s20
// CHECK-NEXT:  	mov	v2.s[3], v3.s[0]
// CHECK-NEXT:  	fadd	v1.4s, v1.4s, v2.4s
// CHECK-NEXT:  	stp	q0, q1, [x2, #32]
// CHECK-NEXT:  	ldp	d9, d8, [sp, #64]
// CHECK-NEXT:  	ldp	d11, d10, [sp, #48]
// CHECK-NEXT:  	ldp	d13, d12, [sp, #32]
// CHECK-NEXT:  	ldp	d15, d14, [sp, #16]
// CHECK-NEXT:  	add	sp, sp, #0x50
// CHECK-NEXT:  	ret
