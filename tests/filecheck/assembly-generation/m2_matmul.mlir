// RUN: mlir-loop %s --no-alias --arch aarch64 --cpu apple-m2 --print-assembly --hide-jumps 2>&1 | filecheck %s

func.func @myfun(
  %A: memref<256x512xf32>,
  %B: memref<512x256xf32>,
  %C: memref<256x256xf32>
) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.fill
    ins(%cst : f32)
    outs(%C : memref<256x256xf32>)
  linalg.matmul
    {
      loop.dims = ["i","j","k"],
      loop.tiles_names = {"i" = ["i1"], "j" = ["j1"], "k" = ["k1"]},
      loop.tiles_sizes = {i1 = 1, j1 = 16, k1 = 4},
      loop.interchange = ["i","j","k","i1","k1","j1"],
      loop.vectorize = ["j1"]
    }
    ins(%A, %B : memref<256x512xf32>, memref<512x256xf32>)
    outs(%C : memref<256x256xf32>)
  return
}
// CHECK:       <myfun>:
// CHECK-NEXT:  	stp	x30, x21, [sp, #-32]!
// CHECK-NEXT:  	stp	x20, x19, [sp, #16]
// CHECK-NEXT:  	mov	x19, x2
// CHECK-NEXT:  	mov	x20, x1
// CHECK-NEXT:  	mov	x21, x0
// CHECK-NEXT:  	mov	x0, x2
// CHECK-NEXT:  	mov	w1, #0x0                   	// #0
// CHECK-NEXT:  	mov	w2, #0x40000               	// #262144
// CHECK-NEXT:  	bl	<memset>
// CHECK-NEXT:  			R_AARCH64_CALL26	memset
// CHECK-NEXT:  	mov	x9, #0x0                   	// #0
// CHECK-NEXT:  	add	x8, x21, #0x8
// CHECK-NEXT:  	mov	x10, #0x0                   	// #0
// CHECK-NEXT:  	add	x11, x19, x9, lsl #10
// CHECK-NEXT:  	mov	x12, x20
// CHECK-NEXT:  	add	x13, x11, x10, lsl #2
// CHECK-NEXT:  	ldp	q1, q0, [x13, #32]
// CHECK-NEXT:  	ldp	q3, q2, [x13]
// CHECK-NEXT:  	mov	x14, #0xfffffffffffffffc    	// #-4
// CHECK-NEXT:  	mov	x15, x8
// CHECK-NEXT:  	mov	x16, x12
// CHECK-NEXT:  	ldur	s4, [x15, #-8]
// CHECK-NEXT:  	prfm	pldl1keep, [x16, #12288]
// CHECK-NEXT:  	ldp	q6, q5, [x16]
// CHECK-NEXT:  	ldp	q7, q16, [x16, #32]
// CHECK-NEXT:  	fmla	v0.4s, v16.4s, v4.s[0]
// CHECK-NEXT:  	fmla	v3.4s, v6.4s, v4.s[0]
// CHECK-NEXT:  	fmla	v2.4s, v5.4s, v4.s[0]
// CHECK-NEXT:  	fmla	v1.4s, v7.4s, v4.s[0]
// CHECK-NEXT:  	ldur	s4, [x15, #-4]
// CHECK-NEXT:  	prfm	pldl1keep, [x16, #13312]
// CHECK-NEXT:  	ldr	q5, [x16, #1072]
// CHECK-NEXT:  	ldr	q6, [x16, #1024]
// CHECK-NEXT:  	ldr	q7, [x16, #1040]
// CHECK-NEXT:  	ldr	q16, [x16, #1056]
// CHECK-NEXT:  	fmla	v1.4s, v16.4s, v4.s[0]
// CHECK-NEXT:  	fmla	v2.4s, v7.4s, v4.s[0]
// CHECK-NEXT:  	fmla	v3.4s, v6.4s, v4.s[0]
// CHECK-NEXT:  	fmla	v0.4s, v5.4s, v4.s[0]
// CHECK-NEXT:  	prfm	pldl1keep, [x16, #14336]
// CHECK-NEXT:  	ldr	q4, [x16, #2080]
// CHECK-NEXT:  	ldr	q5, [x16, #2064]
// CHECK-NEXT:  	ldr	q6, [x16, #2048]
// CHECK-NEXT:  	ldr	q7, [x16, #2096]
// CHECK-NEXT:  	mov	x17, x15
// CHECK-NEXT:  	ld1r	{v16.4s}, [x17], #4
// CHECK-NEXT:  	fmla	v0.4s, v7.4s, v16.4s
// CHECK-NEXT:  	fmla	v3.4s, v6.4s, v16.4s
// CHECK-NEXT:  	fmla	v2.4s, v5.4s, v16.4s
// CHECK-NEXT:  	fmla	v1.4s, v4.4s, v16.4s
// CHECK-NEXT:  	ldr	s4, [x17]
// CHECK-NEXT:  	prfm	pldl1keep, [x16, #15360]
// CHECK-NEXT:  	ldr	q5, [x16, #3120]
// CHECK-NEXT:  	ldr	q6, [x16, #3072]
// CHECK-NEXT:  	ldr	q7, [x16, #3088]
// CHECK-NEXT:  	ldr	q16, [x16, #3104]
// CHECK-NEXT:  	fmla	v1.4s, v16.4s, v4.s[0]
// CHECK-NEXT:  	fmla	v2.4s, v7.4s, v4.s[0]
// CHECK-NEXT:  	fmla	v3.4s, v6.4s, v4.s[0]
// CHECK-NEXT:  	fmla	v0.4s, v5.4s, v4.s[0]
// CHECK-NEXT:  	add	x16, x16, #0x1, lsl #12
// CHECK-NEXT:  	add	x15, x15, #0x10
// CHECK-NEXT:  	add	x14, x14, #0x4
// CHECK-NEXT:  	cmp	x14, #0x1fc
// CHECK-NEXT:  	b.cc	<myfun+0x50>  // b.lo, b.ul, b.last
// CHECK-NEXT:  	stp	q3, q2, [x13]
// CHECK-NEXT:  	stp	q1, q0, [x13, #32]
// CHECK-NEXT:  	add	x13, x10, #0x10
// CHECK-NEXT:  	add	x12, x12, #0x40
// CHECK-NEXT:  	cmp	x10, #0xf0
// CHECK-NEXT:  	mov	x10, x13
// CHECK-NEXT:  	b.cc	<myfun+0x38>  // b.lo, b.ul, b.last
// CHECK-NEXT:  	add	x10, x9, #0x1
// CHECK-NEXT:  	add	x8, x8, #0x800
// CHECK-NEXT:  	cmp	x9, #0xff
// CHECK-NEXT:  	mov	x9, x10
// CHECK-NEXT:  	b.cc	<myfun+0x2c>  // b.lo, b.ul, b.last
// CHECK-NEXT:  	ldp	x20, x19, [sp, #16]
// CHECK-NEXT:  	ldp	x30, x21, [sp], #32
// CHECK-NEXT:  	ret
