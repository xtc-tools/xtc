// RUN: mlir-loop --no-alias --arch aarch64 --cpu apple-m2 --print-assembly --hide-jumps %s 2>&1 | filecheck %s

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
// CHECK-NEXT:  	sub	sp, sp, #0x60
// CHECK-NEXT:  	stp	d15, d14, [sp, #32]
// CHECK-NEXT:  	stp	d13, d12, [sp, #48]
// CHECK-NEXT:  	stp	d11, d10, [sp, #64]
// CHECK-NEXT:  	stp	d9, d8, [sp, #80]
// CHECK-NEXT:  	ldp	s0, s18, [x1]
// CHECK-NEXT:  	ldp	s5, s7, [x0]
// CHECK-NEXT:  	fmul	s6, s5, s0
// CHECK-NEXT:  	ldp	s2, s1, [x1]
// CHECK-NEXT:  	stp	s1, s2, [sp, #8]
// CHECK-NEXT:  	fmul	s4, s5, s18
// CHECK-NEXT:  	ldp	s26, s27, [x1, #8]
// CHECK-NEXT:  	fmul	s16, s5, s26
// CHECK-NEXT:  	mov	v6.s[1], v4.s[0]
// CHECK-NEXT:  	mov	v6.s[2], v16.s[0]
// CHECK-NEXT:  	ldp	s4, s1, [x1, #8]
// CHECK-NEXT:  	fmul	s5, s5, s27
// CHECK-NEXT:  	mov	v6.s[3], v5.s[0]
// CHECK-NEXT:  	ldp	q5, q28, [x2]
// CHECK-NEXT:  	fadd	v20.4s, v5.4s, v6.4s
// CHECK-NEXT:  	ldp	s29, s30, [x1, #16]
// CHECK-NEXT:  	str	s1, [sp, #4]
// CHECK-NEXT:  	fmul	s22, s7, s29
// CHECK-NEXT:  	ldp	s6, s5, [x1, #16]
// CHECK-NEXT:  	fmul	s16, s7, s30
// CHECK-NEXT:  	ldp	s31, s8, [x1, #24]
// CHECK-NEXT:  	fmul	s17, s7, s31
// CHECK-NEXT:  	ldp	s19, s21, [x1, #24]
// CHECK-NEXT:  	fmul	s7, s7, s8
// CHECK-NEXT:  	mov	v22.s[1], v16.s[0]
// CHECK-NEXT:  	mov	v22.s[2], v17.s[0]
// CHECK-NEXT:  	mov	v22.s[3], v7.s[0]
// CHECK-NEXT:  	ldp	s23, s25, [x0, #8]
// CHECK-NEXT:  	ldp	s9, s10, [x1, #32]
// CHECK-NEXT:  	fmul	s24, s23, s9
// CHECK-NEXT:  	ldp	s7, s16, [x1, #32]
// CHECK-NEXT:  	fmul	s12, s23, s10
// CHECK-NEXT:  	ldp	s11, s14, [x1, #40]
// CHECK-NEXT:  	fmul	s13, s23, s11
// CHECK-NEXT:  	fmul	s23, s23, s14
// CHECK-NEXT:  	mov	v24.s[1], v12.s[0]
// CHECK-NEXT:  	mov	v24.s[2], v13.s[0]
// CHECK-NEXT:  	mov	v24.s[3], v23.s[0]
// CHECK-NEXT:  	fadd	v22.4s, v22.4s, v24.4s
// CHECK-NEXT:  	fadd	v12.4s, v20.4s, v22.4s
// CHECK-NEXT:  	ldp	s13, s3, [x1, #48]
// CHECK-NEXT:  	ldp	s17, s20, [x1, #40]
// CHECK-NEXT:  	fmul	s15, s25, s13
// CHECK-NEXT:  	fmul	s23, s25, s3
// CHECK-NEXT:  	mov	v15.s[1], v23.s[0]
// CHECK-NEXT:  	ldp	s2, s1, [x1, #56]
// CHECK-NEXT:  	ldp	s22, s23, [x1, #48]
// CHECK-NEXT:  	fmul	s24, s25, s2
// CHECK-NEXT:  	mov	v15.s[2], v24.s[0]
// CHECK-NEXT:  	ldr	s24, [x1, #56]
// CHECK-NEXT:  	fmul	s25, s25, s1
// CHECK-NEXT:  	mov	v15.s[3], v25.s[0]
// CHECK-NEXT:  	fadd	v25.4s, v12.4s, v15.4s
// CHECK-NEXT:  	str	q25, [sp, #16]
// CHECK-NEXT:  	ldp	s12, s15, [x0, #16]
// CHECK-NEXT:  	fmul	s0, s12, s0
// CHECK-NEXT:  	fmul	s25, s12, s18
// CHECK-NEXT:  	mov	v0.s[1], v25.s[0]
// CHECK-NEXT:  	fmul	s25, s12, s26
// CHECK-NEXT:  	fmul	s26, s12, s27
// CHECK-NEXT:  	mov	v0.s[2], v25.s[0]
// CHECK-NEXT:  	mov	v0.s[3], v26.s[0]
// CHECK-NEXT:  	fadd	v0.4s, v28.4s, v0.4s
// CHECK-NEXT:  	fmul	s25, s15, s29
// CHECK-NEXT:  	fmul	s26, s15, s30
// CHECK-NEXT:  	fmul	s27, s15, s31
// CHECK-NEXT:  	fmul	s28, s15, s8
// CHECK-NEXT:  	mov	v25.s[1], v26.s[0]
// CHECK-NEXT:  	mov	v25.s[2], v27.s[0]
// CHECK-NEXT:  	mov	v25.s[3], v28.s[0]
// CHECK-NEXT:  	ldp	s26, s27, [x0, #24]
// CHECK-NEXT:  	fmul	s28, s26, s9
// CHECK-NEXT:  	fmul	s29, s26, s10
// CHECK-NEXT:  	fmul	s30, s26, s11
// CHECK-NEXT:  	fmul	s26, s26, s14
// CHECK-NEXT:  	mov	v28.s[1], v29.s[0]
// CHECK-NEXT:  	mov	v28.s[2], v30.s[0]
// CHECK-NEXT:  	mov	v28.s[3], v26.s[0]
// CHECK-NEXT:  	fadd	v25.4s, v25.4s, v28.4s
// CHECK-NEXT:  	fadd	v0.4s, v0.4s, v25.4s
// CHECK-NEXT:  	fmul	s25, s27, s13
// CHECK-NEXT:  	fmul	s3, s27, s3
// CHECK-NEXT:  	fmul	s2, s27, s2
// CHECK-NEXT:  	fmul	s1, s27, s1
// CHECK-NEXT:  	mov	v25.s[1], v3.s[0]
// CHECK-NEXT:  	mov	v25.s[2], v2.s[0]
// CHECK-NEXT:  	mov	v25.s[3], v1.s[0]
// CHECK-NEXT:  	fadd	v0.4s, v0.4s, v25.4s
// CHECK-NEXT:  	ldp	s1, s2, [x0, #32]
// CHECK-NEXT:  	ldp	s29, s18, [sp, #8]
// CHECK-NEXT:  	fmul	s3, s1, s18
// CHECK-NEXT:  	fmul	s25, s1, s29
// CHECK-NEXT:  	fmul	s26, s1, s4
// CHECK-NEXT:  	mov	v3.s[1], v25.s[0]
// CHECK-NEXT:  	ldr	s30, [sp, #4]
// CHECK-NEXT:  	fmul	s1, s1, s30
// CHECK-NEXT:  	mov	v3.s[2], v26.s[0]
// CHECK-NEXT:  	mov	v3.s[3], v1.s[0]
// CHECK-NEXT:  	ldp	q25, q1, [x2, #32]
// CHECK-NEXT:  	fadd	v3.4s, v25.4s, v3.4s
// CHECK-NEXT:  	fmul	s25, s2, s6
// CHECK-NEXT:  	fmul	s26, s2, s5
// CHECK-NEXT:  	fmul	s27, s2, s19
// CHECK-NEXT:  	fmul	s2, s2, s21
// CHECK-NEXT:  	mov	v25.s[1], v26.s[0]
// CHECK-NEXT:  	mov	v25.s[2], v27.s[0]
// CHECK-NEXT:  	mov	v25.s[3], v2.s[0]
// CHECK-NEXT:  	ldp	s2, s26, [x0, #40]
// CHECK-NEXT:  	fmul	s27, s2, s7
// CHECK-NEXT:  	fmul	s28, s2, s16
// CHECK-NEXT:  	mov	v27.s[1], v28.s[0]
// CHECK-NEXT:  	fmul	s28, s2, s17
// CHECK-NEXT:  	mov	v27.s[2], v28.s[0]
// CHECK-NEXT:  	fmul	s2, s2, s20
// CHECK-NEXT:  	mov	v27.s[3], v2.s[0]
// CHECK-NEXT:  	fadd	v2.4s, v25.4s, v27.4s
// CHECK-NEXT:  	fmul	s25, s26, s22
// CHECK-NEXT:  	fmul	s27, s26, s23
// CHECK-NEXT:  	mov	v25.s[1], v27.s[0]
// CHECK-NEXT:  	fmul	s27, s26, s24
// CHECK-NEXT:  	mov	v25.s[2], v27.s[0]
// CHECK-NEXT:  	ldr	s27, [x1, #60]
// CHECK-NEXT:  	fmul	s26, s26, s27
// CHECK-NEXT:  	mov	v25.s[3], v26.s[0]
// CHECK-NEXT:  	fadd	v2.4s, v3.4s, v2.4s
// CHECK-NEXT:  	fadd	v26.4s, v2.4s, v25.4s
// CHECK-NEXT:  	ldp	s2, s3, [x0, #48]
// CHECK-NEXT:  	fmul	s25, s2, s18
// CHECK-NEXT:  	fmul	s28, s2, s29
// CHECK-NEXT:  	fmul	s4, s2, s4
// CHECK-NEXT:  	fmul	s2, s2, s30
// CHECK-NEXT:  	mov	v25.s[1], v28.s[0]
// CHECK-NEXT:  	mov	v25.s[2], v4.s[0]
// CHECK-NEXT:  	fmul	s4, s3, s6
// CHECK-NEXT:  	fmul	s5, s3, s5
// CHECK-NEXT:  	fmul	s6, s3, s19
// CHECK-NEXT:  	fmul	s3, s3, s21
// CHECK-NEXT:  	ldp	s18, s19, [x0, #56]
// CHECK-NEXT:  	fmul	s7, s18, s7
// CHECK-NEXT:  	mov	v25.s[3], v2.s[0]
// CHECK-NEXT:  	mov	v4.s[1], v5.s[0]
// CHECK-NEXT:  	fmul	s2, s18, s16
// CHECK-NEXT:  	mov	v4.s[2], v6.s[0]
// CHECK-NEXT:  	mov	v4.s[3], v3.s[0]
// CHECK-NEXT:  	fmul	s3, s18, s17
// CHECK-NEXT:  	fmul	s5, s18, s20
// CHECK-NEXT:  	mov	v7.s[1], v2.s[0]
// CHECK-NEXT:  	mov	v7.s[2], v3.s[0]
// CHECK-NEXT:  	mov	v7.s[3], v5.s[0]
// CHECK-NEXT:  	fadd	v1.4s, v1.4s, v25.4s
// CHECK-NEXT:  	fadd	v2.4s, v4.4s, v7.4s
// CHECK-NEXT:  	fadd	v1.4s, v1.4s, v2.4s
// CHECK-NEXT:  	ldr	q2, [sp, #16]
// CHECK-NEXT:  	stp	q2, q0, [x2]
// CHECK-NEXT:  	fmul	s0, s19, s22
// CHECK-NEXT:  	fmul	s2, s19, s23
// CHECK-NEXT:  	fmul	s3, s19, s24
// CHECK-NEXT:  	mov	v0.s[1], v2.s[0]
// CHECK-NEXT:  	mov	v0.s[2], v3.s[0]
// CHECK-NEXT:  	fmul	s2, s19, s27
// CHECK-NEXT:  	mov	v0.s[3], v2.s[0]
// CHECK-NEXT:  	fadd	v0.4s, v1.4s, v0.4s
// CHECK-NEXT:  	stp	q26, q0, [x2, #32]
// CHECK-NEXT:  	ldp	d9, d8, [sp, #80]
// CHECK-NEXT:  	ldp	d11, d10, [sp, #64]
// CHECK-NEXT:  	ldp	d13, d12, [sp, #48]
// CHECK-NEXT:  	ldp	d15, d14, [sp, #32]
// CHECK-NEXT:  	add	sp, sp, #0x60
// CHECK-NEXT:  	ret
