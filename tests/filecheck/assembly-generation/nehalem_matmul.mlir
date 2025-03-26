// RUN: mlir-loop %s --no-alias --always-vectorize --arch x86-64 --cpu nehalem --print-assembly --hide-jumps 2>&1 | filecheck %s

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
      loop.tiles = {"i" = {"i1" = 1}, "j" = {"j1" = 16}, "k" = {"k1" = 4}},
      loop.interchange = ["i","j","k","i1","k1","j1"],
      loop.vectorize = ["j1"],
      loop.unroll = {i1 = 1, k1 = 4}
    }
    ins(%A, %B : memref<256x512xf32>, memref<512x256xf32>)
    outs(%C : memref<256x256xf32>)
  return
}
// CHECK:       <myfun>:
// CHECK-NEXT:  	push   %r15
// CHECK-NEXT:  	push   %r14
// CHECK-NEXT:  	push   %r12
// CHECK-NEXT:  	push   %rbx
// CHECK-NEXT:  	push   %rax
// CHECK-NEXT:  	mov    %rdx,%rbx
// CHECK-NEXT:  	mov    %rsi,%r14
// CHECK-NEXT:  	mov    %rdi,%r15
// CHECK-NEXT:  	xor    %r12d,%r12d
// CHECK-NEXT:  	mov    $0x40000,%edx
// CHECK-NEXT:  	mov    %rbx,%rdi
// CHECK-NEXT:  	xor    %esi,%esi
// CHECK-NEXT:  	call   <myfun+0x23>
// CHECK-NEXT:  			R_X86_64_PLT32	memset-0x4
// CHECK-NEXT:  	add    $0xc00,%r14
// CHECK-NEXT:  	add    $0xc,%r15
// CHECK-NEXT:  	xchg   %ax,%ax
// CHECK-NEXT:  	mov    %r12,%rax
// CHECK-NEXT:  	shl    $0xa,%rax
// CHECK-NEXT:  	add    %rbx,%rax
// CHECK-NEXT:  	mov    %r14,%rcx
// CHECK-NEXT:  	xor    %edx,%edx
// CHECK-NEXT:  	nop
// CHECK-NEXT:  	movups (%rax,%rdx,4),%xmm0
// CHECK-NEXT:  	movups 0x10(%rax,%rdx,4),%xmm4
// CHECK-NEXT:  	movups 0x20(%rax,%rdx,4),%xmm3
// CHECK-NEXT:  	movups 0x30(%rax,%rdx,4),%xmm1
// CHECK-NEXT:  	mov    $0xfffffffffffffffc,%rsi
// CHECK-NEXT:  	mov    %rcx,%rdi
// CHECK-NEXT:  	nopl   (%rax)
// CHECK-NEXT:  	movups -0xc00(%rdi),%xmm6
// CHECK-NEXT:  	movups -0xbf0(%rdi),%xmm7
// CHECK-NEXT:  	movups -0xbe0(%rdi),%xmm8
// CHECK-NEXT:  	movups -0xbd0(%rdi),%xmm9
// CHECK-NEXT:  	movss  0x4(%r15,%rsi,4),%xmm10
// CHECK-NEXT:  	movss  0x8(%r15,%rsi,4),%xmm5
// CHECK-NEXT:  	movss  0xc(%r15,%rsi,4),%xmm2
// CHECK-NEXT:  	shufps $0x0,%xmm10,%xmm10
// CHECK-NEXT:  	mulps  %xmm10,%xmm6
// CHECK-NEXT:  	addps  %xmm0,%xmm6
// CHECK-NEXT:  	movss  0x10(%r15,%rsi,4),%xmm0
// CHECK-NEXT:  	mulps  %xmm10,%xmm7
// CHECK-NEXT:  	addps  %xmm4,%xmm7
// CHECK-NEXT:  	mulps  %xmm10,%xmm8
// CHECK-NEXT:  	addps  %xmm3,%xmm8
// CHECK-NEXT:  	mulps  %xmm9,%xmm10
// CHECK-NEXT:  	addps  %xmm1,%xmm10
// CHECK-NEXT:  	movups -0x800(%rdi),%xmm9
// CHECK-NEXT:  	movups -0x7f0(%rdi),%xmm4
// CHECK-NEXT:  	movups -0x7e0(%rdi),%xmm3
// CHECK-NEXT:  	movups -0x7d0(%rdi),%xmm1
// CHECK-NEXT:  	shufps $0x0,%xmm5,%xmm5
// CHECK-NEXT:  	mulps  %xmm5,%xmm1
// CHECK-NEXT:  	addps  %xmm10,%xmm1
// CHECK-NEXT:  	mulps  %xmm5,%xmm3
// CHECK-NEXT:  	addps  %xmm8,%xmm3
// CHECK-NEXT:  	mulps  %xmm5,%xmm4
// CHECK-NEXT:  	addps  %xmm7,%xmm4
// CHECK-NEXT:  	mulps  %xmm9,%xmm5
// CHECK-NEXT:  	addps  %xmm6,%xmm5
// CHECK-NEXT:  	movups -0x3d0(%rdi),%xmm9
// CHECK-NEXT:  	movups -0x3e0(%rdi),%xmm8
// CHECK-NEXT:  	movups -0x3f0(%rdi),%xmm7
// CHECK-NEXT:  	movups -0x400(%rdi),%xmm6
// CHECK-NEXT:  	shufps $0x0,%xmm2,%xmm2
// CHECK-NEXT:  	mulps  %xmm2,%xmm6
// CHECK-NEXT:  	addps  %xmm5,%xmm6
// CHECK-NEXT:  	mulps  %xmm2,%xmm7
// CHECK-NEXT:  	addps  %xmm4,%xmm7
// CHECK-NEXT:  	mulps  %xmm2,%xmm8
// CHECK-NEXT:  	addps  %xmm3,%xmm8
// CHECK-NEXT:  	mulps  %xmm9,%xmm2
// CHECK-NEXT:  	addps  %xmm1,%xmm2
// CHECK-NEXT:  	movups (%rdi),%xmm5
// CHECK-NEXT:  	movups 0x10(%rdi),%xmm4
// CHECK-NEXT:  	movups 0x20(%rdi),%xmm3
// CHECK-NEXT:  	movups 0x30(%rdi),%xmm1
// CHECK-NEXT:  	shufps $0x0,%xmm0,%xmm0
// CHECK-NEXT:  	mulps  %xmm0,%xmm1
// CHECK-NEXT:  	addps  %xmm2,%xmm1
// CHECK-NEXT:  	mulps  %xmm0,%xmm3
// CHECK-NEXT:  	addps  %xmm8,%xmm3
// CHECK-NEXT:  	mulps  %xmm0,%xmm4
// CHECK-NEXT:  	addps  %xmm7,%xmm4
// CHECK-NEXT:  	mulps  %xmm5,%xmm0
// CHECK-NEXT:  	addps  %xmm6,%xmm0
// CHECK-NEXT:  	add    $0x4,%rsi
// CHECK-NEXT:  	add    $0x1000,%rdi
// CHECK-NEXT:  	cmp    $0x1fc,%rsi
// CHECK-NEXT:  	jb     <myfun+0x60>
// CHECK-NEXT:  	movups %xmm0,(%rax,%rdx,4)
// CHECK-NEXT:  	movups %xmm4,0x10(%rax,%rdx,4)
// CHECK-NEXT:  	movups %xmm3,0x20(%rax,%rdx,4)
// CHECK-NEXT:  	movups %xmm1,0x30(%rax,%rdx,4)
// CHECK-NEXT:  	lea    0x10(%rdx),%rsi
// CHECK-NEXT:  	add    $0x40,%rcx
// CHECK-NEXT:  	cmp    $0xf0,%rdx
// CHECK-NEXT:  	mov    %rsi,%rdx
// CHECK-NEXT:  	jb     <myfun+0x40>
// CHECK-NEXT:  	lea    0x1(%r12),%rax
// CHECK-NEXT:  	add    $0x800,%r15
// CHECK-NEXT:  	cmp    $0xff,%r12
// CHECK-NEXT:  	mov    %rax,%r12
// CHECK-NEXT:  	jb     <myfun+0x30>
// CHECK-NEXT:  	add    $0x8,%rsp
// CHECK-NEXT:  	pop    %rbx
// CHECK-NEXT:  	pop    %r12
// CHECK-NEXT:  	pop    %r14
// CHECK-NEXT:  	pop    %r15
// CHECK-NEXT:  	ret
