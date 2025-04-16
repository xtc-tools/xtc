// RUN: mlir-loop %s --no-alias --arch aarch64 --cpu cortex-a72 --print-assembly | filecheck %s

func.func @myfun(
  %A: memref<4x4xf32>,
  %B: memref<4x4xf32>,
  %C: memref<4x4xf32>
) {
  linalg.matmul
    ins(%A, %B : memref<4x4xf32>, memref<4x4xf32>)
    outs(%C : memref<4x4xf32>)
  return
}
// CHECK:       <myfun>:
// CHECK-NEXT:  	stp	d13, d12, [sp, #-48]!
// CHECK-NEXT:  	stp	d11, d10, [sp, #16]
// CHECK-NEXT:  	stp	d9, d8, [sp, #32]
// CHECK-NEXT:  	ldp	s4, s0, [x1]
// CHECK-NEXT:  	ldp	s20, s24, [x0]
// CHECK-NEXT:  	fmul	s16, s20, s4
// CHECK-NEXT:  	ldp	s7, s2, [x1, #16]
// CHECK-NEXT:  	fmul	s19, s24, s7
// CHECK-NEXT:  	ldp	s6, s3, [x1, #32]
// CHECK-NEXT:  	ldp	s25, s26, [x0, #8]
// CHECK-NEXT:  	fmul	s27, s25, s6
// CHECK-NEXT:  	ldp	s18, s5, [x1, #48]
// CHECK-NEXT:  	fmul	s28, s26, s18
// CHECK-NEXT:  	fmul	s21, s20, s0
// CHECK-NEXT:  	fmul	s23, s24, s2
// CHECK-NEXT:  	fmul	s29, s25, s3
// CHECK-NEXT:  	fmul	s30, s26, s5
// CHECK-NEXT:  	ldp	s17, s1, [x1, #8]
// CHECK-NEXT:  	fmul	s22, s20, s17
// CHECK-NEXT:  	ldp	s31, s8, [x2]
// CHECK-NEXT:  	fadd	s31, s31, s16
// CHECK-NEXT:  	fadd	s8, s8, s21
// CHECK-NEXT:  	ldp	s16, s9, [x2, #8]
// CHECK-NEXT:  	fadd	s10, s16, s22
// CHECK-NEXT:  	ldp	s21, s16, [x1, #24]
// CHECK-NEXT:  	fmul	s11, s24, s21
// CHECK-NEXT:  	fadd	s31, s31, s19
// CHECK-NEXT:  	ldp	s22, s19, [x1, #40]
// CHECK-NEXT:  	fmul	s12, s25, s22
// CHECK-NEXT:  	fmul	s20, s20, s1
// CHECK-NEXT:  	fadd	s9, s9, s20
// CHECK-NEXT:  	fadd	s8, s8, s23
// CHECK-NEXT:  	fadd	s10, s10, s11
// CHECK-NEXT:  	ldp	s23, s20, [x1, #56]
// CHECK-NEXT:  	fmul	s11, s26, s23
// CHECK-NEXT:  	fmul	s24, s24, s16
// CHECK-NEXT:  	fmul	s25, s25, s19
// CHECK-NEXT:  	fmul	s13, s26, s20
// CHECK-NEXT:  	fadd	s24, s9, s24
// CHECK-NEXT:  	fadd	s26, s31, s27
// CHECK-NEXT:  	fadd	s27, s8, s29
// CHECK-NEXT:  	ldp	s29, s31, [x0, #16]
// CHECK-NEXT:  	fmul	s8, s29, s4
// CHECK-NEXT:  	fadd	s9, s10, s12
// CHECK-NEXT:  	fadd	s10, s24, s25
// CHECK-NEXT:  	fadd	s24, s26, s28
// CHECK-NEXT:  	fadd	s25, s27, s30
// CHECK-NEXT:  	fadd	s26, s9, s11
// CHECK-NEXT:  	fadd	s27, s10, s13
// CHECK-NEXT:  	ldp	s28, s30, [x2, #16]
// CHECK-NEXT:  	fadd	s28, s28, s8
// CHECK-NEXT:  	fmul	s8, s31, s7
// CHECK-NEXT:  	fadd	s28, s28, s8
// CHECK-NEXT:  	ldp	s8, s9, [x0, #24]
// CHECK-NEXT:  	fmul	s10, s8, s6
// CHECK-NEXT:  	fadd	s28, s28, s10
// CHECK-NEXT:  	fmul	s10, s9, s18
// CHECK-NEXT:  	fadd	s28, s28, s10
// CHECK-NEXT:  	fmul	s10, s29, s0
// CHECK-NEXT:  	fadd	s30, s30, s10
// CHECK-NEXT:  	fmul	s10, s31, s2
// CHECK-NEXT:  	fadd	s30, s30, s10
// CHECK-NEXT:  	fmul	s10, s8, s3
// CHECK-NEXT:  	fadd	s30, s30, s10
// CHECK-NEXT:  	fmul	s10, s9, s5
// CHECK-NEXT:  	fadd	s30, s30, s10
// CHECK-NEXT:  	fmul	s10, s29, s17
// CHECK-NEXT:  	ldp	s11, s12, [x2, #24]
// CHECK-NEXT:  	fadd	s10, s11, s10
// CHECK-NEXT:  	fmul	s11, s31, s21
// CHECK-NEXT:  	fadd	s10, s10, s11
// CHECK-NEXT:  	fmul	s11, s8, s22
// CHECK-NEXT:  	fadd	s10, s10, s11
// CHECK-NEXT:  	fmul	s11, s9, s23
// CHECK-NEXT:  	fadd	s10, s10, s11
// CHECK-NEXT:  	fmul	s29, s29, s1
// CHECK-NEXT:  	fadd	s29, s12, s29
// CHECK-NEXT:  	fmul	s31, s31, s16
// CHECK-NEXT:  	fadd	s29, s29, s31
// CHECK-NEXT:  	fmul	s31, s8, s19
// CHECK-NEXT:  	fadd	s29, s29, s31
// CHECK-NEXT:  	fmul	s31, s9, s20
// CHECK-NEXT:  	fadd	s29, s29, s31
// CHECK-NEXT:  	ldp	s31, s8, [x0, #32]
// CHECK-NEXT:  	fmul	s9, s31, s4
// CHECK-NEXT:  	ldp	s11, s12, [x2, #32]
// CHECK-NEXT:  	fadd	s9, s11, s9
// CHECK-NEXT:  	stp	s24, s25, [x2]
// CHECK-NEXT:  	stp	s26, s27, [x2, #8]
// CHECK-NEXT:  	stp	s28, s30, [x2, #16]
// CHECK-NEXT:  	stp	s10, s29, [x2, #24]
// CHECK-NEXT:  	fmul	s24, s8, s7
// CHECK-NEXT:  	fadd	s24, s9, s24
// CHECK-NEXT:  	ldp	s25, s26, [x0, #40]
// CHECK-NEXT:  	fmul	s27, s25, s6
// CHECK-NEXT:  	fadd	s24, s24, s27
// CHECK-NEXT:  	fmul	s27, s26, s18
// CHECK-NEXT:  	fadd	s24, s24, s27
// CHECK-NEXT:  	fmul	s27, s31, s0
// CHECK-NEXT:  	fadd	s27, s12, s27
// CHECK-NEXT:  	fmul	s28, s8, s2
// CHECK-NEXT:  	fadd	s27, s27, s28
// CHECK-NEXT:  	fmul	s28, s25, s3
// CHECK-NEXT:  	fadd	s27, s27, s28
// CHECK-NEXT:  	fmul	s28, s26, s5
// CHECK-NEXT:  	fadd	s27, s27, s28
// CHECK-NEXT:  	stp	s24, s27, [x2, #32]
// CHECK-NEXT:  	fmul	s24, s31, s17
// CHECK-NEXT:  	ldp	s27, s28, [x2, #40]
// CHECK-NEXT:  	fadd	s24, s27, s24
// CHECK-NEXT:  	fmul	s27, s8, s21
// CHECK-NEXT:  	fadd	s24, s24, s27
// CHECK-NEXT:  	fmul	s27, s25, s22
// CHECK-NEXT:  	fadd	s24, s24, s27
// CHECK-NEXT:  	fmul	s27, s26, s23
// CHECK-NEXT:  	fadd	s24, s24, s27
// CHECK-NEXT:  	fmul	s27, s31, s1
// CHECK-NEXT:  	fadd	s27, s28, s27
// CHECK-NEXT:  	fmul	s28, s8, s16
// CHECK-NEXT:  	fadd	s27, s27, s28
// CHECK-NEXT:  	fmul	s25, s25, s19
// CHECK-NEXT:  	fadd	s25, s27, s25
// CHECK-NEXT:  	fmul	s26, s26, s20
// CHECK-NEXT:  	fadd	s25, s25, s26
// CHECK-NEXT:  	stp	s24, s25, [x2, #40]
// CHECK-NEXT:  	ldp	s24, s25, [x0, #48]
// CHECK-NEXT:  	fmul	s4, s24, s4
// CHECK-NEXT:  	ldp	s26, s27, [x2, #48]
// CHECK-NEXT:  	fadd	s4, s26, s4
// CHECK-NEXT:  	fmul	s7, s25, s7
// CHECK-NEXT:  	fadd	s4, s4, s7
// CHECK-NEXT:  	ldp	s7, s26, [x0, #56]
// CHECK-NEXT:  	fmul	s6, s7, s6
// CHECK-NEXT:  	fadd	s4, s4, s6
// CHECK-NEXT:  	fmul	s6, s26, s18
// CHECK-NEXT:  	fadd	s4, s4, s6
// CHECK-NEXT:  	fmul	s0, s24, s0
// CHECK-NEXT:  	fadd	s0, s27, s0
// CHECK-NEXT:  	fmul	s2, s25, s2
// CHECK-NEXT:  	fadd	s0, s0, s2
// CHECK-NEXT:  	fmul	s2, s7, s3
// CHECK-NEXT:  	fadd	s0, s0, s2
// CHECK-NEXT:  	fmul	s2, s26, s5
// CHECK-NEXT:  	fadd	s0, s0, s2
// CHECK-NEXT:  	stp	s4, s0, [x2, #48]
// CHECK-NEXT:  	fmul	s0, s24, s17
// CHECK-NEXT:  	ldp	s2, s3, [x2, #56]
// CHECK-NEXT:  	fadd	s0, s2, s0
// CHECK-NEXT:  	fmul	s2, s25, s21
// CHECK-NEXT:  	fadd	s0, s0, s2
// CHECK-NEXT:  	fmul	s2, s7, s22
// CHECK-NEXT:  	fadd	s0, s0, s2
// CHECK-NEXT:  	fmul	s2, s26, s23
// CHECK-NEXT:  	fadd	s0, s0, s2
// CHECK-NEXT:  	fmul	s1, s24, s1
// CHECK-NEXT:  	fadd	s1, s3, s1
// CHECK-NEXT:  	fmul	s2, s25, s16
// CHECK-NEXT:  	fadd	s1, s1, s2
// CHECK-NEXT:  	fmul	s2, s7, s19
// CHECK-NEXT:  	fadd	s1, s1, s2
// CHECK-NEXT:  	fmul	s2, s26, s20
// CHECK-NEXT:  	fadd	s1, s1, s2
// CHECK-NEXT:  	stp	s0, s1, [x2, #56]
// CHECK-NEXT:  	ldp	d9, d8, [sp, #32]
// CHECK-NEXT:  	ldp	d11, d10, [sp, #16]
// CHECK-NEXT:  	ldp	d13, d12, [sp], #48
// CHECK-NEXT:  	ret
