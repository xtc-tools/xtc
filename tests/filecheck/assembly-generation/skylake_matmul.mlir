// RUN: mlir-loop %s --no-alias --arch x86-64 --cpu skylake --print-assembly --hide-jumps 2>&1 | filecheck %s

func.func @myfun(
  %A: memref<256x512xf32>,
  %B: memref<512x256xf32>,
  %C: memref<256x256xf32>
) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.fill
      {
        loop.dims = ["i","j"],
        loop.tiles_names = {"i" = ["i1"], "j" = ["j1"]},
        loop.tiles_sizes = {i1 = 1, j1 = 64},
        loop.interchange = ["i","j","i1","j1"],
        loop.vectorize = ["j1"]
    }
    ins(%cst : f32)
    outs(%C : memref<256x256xf32>)
  linalg.matmul
    {
      loop.dims = ["i","j","k"],
      loop.tiles_names = {"i" = ["i1"], "j" = ["j1"], "k" = ["k1"]},
      loop.tiles_sizes = {i1 = 1, j1 = 64, k1 = 8},
      loop.interchange = ["i","j","k","i1","k1","j1"],
      loop.vectorize = ["j1"]
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
// CHECK-NEXT:  	add    $0x1c,%r15
// CHECK-NEXT:  	add    $0x1c00,%r14
// CHECK-NEXT:  	xchg   %ax,%ax
// CHECK-NEXT:  	mov    %r12,%rax
// CHECK-NEXT:  	shl    $0xa,%rax
// CHECK-NEXT:  	add    %rbx,%rax
// CHECK-NEXT:  	mov    %r14,%rcx
// CHECK-NEXT:  	xor    %edx,%edx
// CHECK-NEXT:  	nop
// CHECK-NEXT:  	vmovups 0xe0(%rax,%rdx,4),%ymm0
// CHECK-NEXT:  	vmovups 0xc0(%rax,%rdx,4),%ymm1
// CHECK-NEXT:  	vmovups 0xa0(%rax,%rdx,4),%ymm2
// CHECK-NEXT:  	vmovups 0x80(%rax,%rdx,4),%ymm3
// CHECK-NEXT:  	vmovups (%rax,%rdx,4),%ymm4
// CHECK-NEXT:  	vmovups 0x20(%rax,%rdx,4),%ymm5
// CHECK-NEXT:  	vmovups 0x40(%rax,%rdx,4),%ymm6
// CHECK-NEXT:  	vmovups 0x60(%rax,%rdx,4),%ymm7
// CHECK-NEXT:  	mov    $0xfffffffffffffff8,%rsi
// CHECK-NEXT:  	mov    %rcx,%rdi
// CHECK-NEXT:  	data16 cs nopw 0x0(%rax,%rax,1)
// CHECK-NEXT:  	vbroadcastss 0x4(%r15,%rsi,4),%ymm8
// CHECK-NEXT:  	vfmadd231ps -0x1b20(%rdi),%ymm8,%ymm0
// CHECK-NEXT:  	vfmadd231ps -0x1b40(%rdi),%ymm8,%ymm1
// CHECK-NEXT:  	vfmadd231ps -0x1b60(%rdi),%ymm8,%ymm2
// CHECK-NEXT:  	vfmadd231ps -0x1b80(%rdi),%ymm8,%ymm3
// CHECK-NEXT:  	vfmadd231ps -0x1ba0(%rdi),%ymm8,%ymm7
// CHECK-NEXT:  	vfmadd231ps -0x1bc0(%rdi),%ymm8,%ymm6
// CHECK-NEXT:  	vfmadd231ps -0x1c00(%rdi),%ymm8,%ymm4
// CHECK-NEXT:  	vfmadd231ps -0x1be0(%rdi),%ymm8,%ymm5
// CHECK-NEXT:  	vbroadcastss 0x8(%r15,%rsi,4),%ymm8
// CHECK-NEXT:  	vfmadd231ps -0x17e0(%rdi),%ymm8,%ymm5
// CHECK-NEXT:  	vfmadd231ps -0x1800(%rdi),%ymm8,%ymm4
// CHECK-NEXT:  	vfmadd231ps -0x17c0(%rdi),%ymm8,%ymm6
// CHECK-NEXT:  	vfmadd231ps -0x17a0(%rdi),%ymm8,%ymm7
// CHECK-NEXT:  	vfmadd231ps -0x1780(%rdi),%ymm8,%ymm3
// CHECK-NEXT:  	vfmadd231ps -0x1760(%rdi),%ymm8,%ymm2
// CHECK-NEXT:  	vfmadd231ps -0x1740(%rdi),%ymm8,%ymm1
// CHECK-NEXT:  	vfmadd231ps -0x1720(%rdi),%ymm8,%ymm0
// CHECK-NEXT:  	vbroadcastss 0xc(%r15,%rsi,4),%ymm8
// CHECK-NEXT:  	vfmadd231ps -0x1320(%rdi),%ymm8,%ymm0
// CHECK-NEXT:  	vfmadd231ps -0x1340(%rdi),%ymm8,%ymm1
// CHECK-NEXT:  	vfmadd231ps -0x1360(%rdi),%ymm8,%ymm2
// CHECK-NEXT:  	vfmadd231ps -0x1380(%rdi),%ymm8,%ymm3
// CHECK-NEXT:  	vfmadd231ps -0x13a0(%rdi),%ymm8,%ymm7
// CHECK-NEXT:  	vfmadd231ps -0x13c0(%rdi),%ymm8,%ymm6
// CHECK-NEXT:  	vfmadd231ps -0x1400(%rdi),%ymm8,%ymm4
// CHECK-NEXT:  	vfmadd231ps -0x13e0(%rdi),%ymm8,%ymm5
// CHECK-NEXT:  	vbroadcastss 0x10(%r15,%rsi,4),%ymm8
// CHECK-NEXT:  	vfmadd231ps -0xfe0(%rdi),%ymm8,%ymm5
// CHECK-NEXT:  	vfmadd231ps -0x1000(%rdi),%ymm8,%ymm4
// CHECK-NEXT:  	vfmadd231ps -0xfc0(%rdi),%ymm8,%ymm6
// CHECK-NEXT:  	vfmadd231ps -0xfa0(%rdi),%ymm8,%ymm7
// CHECK-NEXT:  	vfmadd231ps -0xf80(%rdi),%ymm8,%ymm3
// CHECK-NEXT:  	vfmadd231ps -0xf60(%rdi),%ymm8,%ymm2
// CHECK-NEXT:  	vfmadd231ps -0xf40(%rdi),%ymm8,%ymm1
// CHECK-NEXT:  	vfmadd231ps -0xf20(%rdi),%ymm8,%ymm0
// CHECK-NEXT:  	vbroadcastss 0x14(%r15,%rsi,4),%ymm8
// CHECK-NEXT:  	vfmadd231ps -0xb20(%rdi),%ymm8,%ymm0
// CHECK-NEXT:  	vfmadd231ps -0xb40(%rdi),%ymm8,%ymm1
// CHECK-NEXT:  	vfmadd231ps -0xb60(%rdi),%ymm8,%ymm2
// CHECK-NEXT:  	vfmadd231ps -0xb80(%rdi),%ymm8,%ymm3
// CHECK-NEXT:  	vfmadd231ps -0xba0(%rdi),%ymm8,%ymm7
// CHECK-NEXT:  	vfmadd231ps -0xbc0(%rdi),%ymm8,%ymm6
// CHECK-NEXT:  	vfmadd231ps -0xc00(%rdi),%ymm8,%ymm4
// CHECK-NEXT:  	vfmadd231ps -0xbe0(%rdi),%ymm8,%ymm5
// CHECK-NEXT:  	vbroadcastss 0x18(%r15,%rsi,4),%ymm8
// CHECK-NEXT:  	vfmadd231ps -0x7e0(%rdi),%ymm8,%ymm5
// CHECK-NEXT:  	vfmadd231ps -0x800(%rdi),%ymm8,%ymm4
// CHECK-NEXT:  	vfmadd231ps -0x7c0(%rdi),%ymm8,%ymm6
// CHECK-NEXT:  	vfmadd231ps -0x7a0(%rdi),%ymm8,%ymm7
// CHECK-NEXT:  	vfmadd231ps -0x780(%rdi),%ymm8,%ymm3
// CHECK-NEXT:  	vfmadd231ps -0x760(%rdi),%ymm8,%ymm2
// CHECK-NEXT:  	vfmadd231ps -0x740(%rdi),%ymm8,%ymm1
// CHECK-NEXT:  	vfmadd231ps -0x720(%rdi),%ymm8,%ymm0
// CHECK-NEXT:  	vbroadcastss 0x1c(%r15,%rsi,4),%ymm8
// CHECK-NEXT:  	vfmadd231ps -0x320(%rdi),%ymm8,%ymm0
// CHECK-NEXT:  	vfmadd231ps -0x340(%rdi),%ymm8,%ymm1
// CHECK-NEXT:  	vfmadd231ps -0x360(%rdi),%ymm8,%ymm2
// CHECK-NEXT:  	vfmadd231ps -0x380(%rdi),%ymm8,%ymm3
// CHECK-NEXT:  	vfmadd231ps -0x3a0(%rdi),%ymm8,%ymm7
// CHECK-NEXT:  	vfmadd231ps -0x3c0(%rdi),%ymm8,%ymm6
// CHECK-NEXT:  	vfmadd231ps -0x400(%rdi),%ymm8,%ymm4
// CHECK-NEXT:  	vfmadd231ps -0x3e0(%rdi),%ymm8,%ymm5
// CHECK-NEXT:  	vbroadcastss 0x20(%r15,%rsi,4),%ymm8
// CHECK-NEXT:  	vfmadd231ps 0x20(%rdi),%ymm8,%ymm5
// CHECK-NEXT:  	vfmadd231ps (%rdi),%ymm8,%ymm4
// CHECK-NEXT:  	vfmadd231ps 0x40(%rdi),%ymm8,%ymm6
// CHECK-NEXT:  	vfmadd231ps 0x60(%rdi),%ymm8,%ymm7
// CHECK-NEXT:  	vfmadd231ps 0x80(%rdi),%ymm8,%ymm3
// CHECK-NEXT:  	vfmadd231ps 0xa0(%rdi),%ymm8,%ymm2
// CHECK-NEXT:  	vfmadd231ps 0xc0(%rdi),%ymm8,%ymm1
// CHECK-NEXT:  	vfmadd231ps 0xe0(%rdi),%ymm8,%ymm0
// CHECK-NEXT:  	add    $0x8,%rsi
// CHECK-NEXT:  	add    $0x2000,%rdi
// CHECK-NEXT:  	cmp    $0x1f8,%rsi
// CHECK-NEXT:  	jb     <myfun+0x90>
// CHECK-NEXT:  	vmovups %ymm4,(%rax,%rdx,4)
// CHECK-NEXT:  	vmovups %ymm5,0x20(%rax,%rdx,4)
// CHECK-NEXT:  	vmovups %ymm6,0x40(%rax,%rdx,4)
// CHECK-NEXT:  	vmovups %ymm7,0x60(%rax,%rdx,4)
// CHECK-NEXT:  	vmovups %ymm3,0x80(%rax,%rdx,4)
// CHECK-NEXT:  	vmovups %ymm2,0xa0(%rax,%rdx,4)
// CHECK-NEXT:  	vmovups %ymm1,0xc0(%rax,%rdx,4)
// CHECK-NEXT:  	vmovups %ymm0,0xe0(%rax,%rdx,4)
// CHECK-NEXT:  	lea    0x40(%rdx),%rsi
// CHECK-NEXT:  	add    $0x100,%rcx
// CHECK-NEXT:  	cmp    $0xc0,%rdx
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
// CHECK-NEXT:  	vzeroupper
// CHECK-NEXT:  	ret
