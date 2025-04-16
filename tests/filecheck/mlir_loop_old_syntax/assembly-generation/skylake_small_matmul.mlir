// RUN: mlir-loop --old-syntax %s --no-alias --arch x86-64 --cpu skylake --print-assembly --hide-jumps 2>&1 | filecheck %s

func.func @myfun(
  %A: memref<16x4xf32>,
  %B: memref<4x8xf32>,
  %C: memref<16x8xf32>
) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.fill
    ins(%cst : f32)
    outs(%C : memref<16x8xf32>)
  linalg.matmul
    {
      loop.dims = ["i","j","k"],
      loop.interchange = ["i","k","j"],
      loop.vectorize = ["j"]
    }
    ins(%A, %B : memref<16x4xf32>, memref<4x8xf32>)
    outs(%C : memref<16x8xf32>)
  return
}
// CHECK:       <myfun>:
// CHECK-NEXT:  	vxorps %xmm0,%xmm0,%xmm0
// CHECK-NEXT:  	vmovups %ymm0,0x1e0(%rdx)
// CHECK-NEXT:  	vmovups %ymm0,0x1c0(%rdx)
// CHECK-NEXT:  	vmovups %ymm0,0x1a0(%rdx)
// CHECK-NEXT:  	vmovups %ymm0,0x180(%rdx)
// CHECK-NEXT:  	vmovups %ymm0,0x160(%rdx)
// CHECK-NEXT:  	vmovups %ymm0,0x140(%rdx)
// CHECK-NEXT:  	vmovups %ymm0,0x120(%rdx)
// CHECK-NEXT:  	vmovups %ymm0,0x100(%rdx)
// CHECK-NEXT:  	vmovups %ymm0,0xe0(%rdx)
// CHECK-NEXT:  	vmovups %ymm0,0xc0(%rdx)
// CHECK-NEXT:  	vmovups %ymm0,0xa0(%rdx)
// CHECK-NEXT:  	vmovups %ymm0,0x80(%rdx)
// CHECK-NEXT:  	vmovups %ymm0,0x60(%rdx)
// CHECK-NEXT:  	vmovups %ymm0,0x40(%rdx)
// CHECK-NEXT:  	vmovups %ymm0,0x20(%rdx)
// CHECK-NEXT:  	vmovups %ymm0,(%rdx)
// CHECK-NEXT:  	vmovups (%rsi),%ymm0
// CHECK-NEXT:  	vmovups 0x20(%rsi),%ymm1
// CHECK-NEXT:  	vmovups 0x40(%rsi),%ymm2
// CHECK-NEXT:  	vmovups 0x60(%rsi),%ymm3
// CHECK-NEXT:  	mov    $0xffffffffffffffff,%rax
// CHECK-NEXT:  	xor    %ecx,%ecx
// CHECK-NEXT:  	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
// CHECK-NEXT:  	vbroadcastss (%rdi,%rcx,1),%ymm4
// CHECK-NEXT:  	vfmadd213ps (%rdx,%rcx,2),%ymm0,%ymm4
// CHECK-NEXT:  	vbroadcastss 0x4(%rdi,%rcx,1),%ymm5
// CHECK-NEXT:  	vfmadd213ps %ymm4,%ymm1,%ymm5
// CHECK-NEXT:  	vbroadcastss 0x8(%rdi,%rcx,1),%ymm4
// CHECK-NEXT:  	vfmadd213ps %ymm5,%ymm2,%ymm4
// CHECK-NEXT:  	vbroadcastss 0xc(%rdi,%rcx,1),%ymm5
// CHECK-NEXT:  	vfmadd213ps %ymm4,%ymm3,%ymm5
// CHECK-NEXT:  	vmovups %ymm5,(%rdx,%rcx,2)
// CHECK-NEXT:  	inc    %rax
// CHECK-NEXT:  	add    $0x10,%rcx
// CHECK-NEXT:  	cmp    $0xf,%rax
// CHECK-NEXT:  	jb     <myfun+0xa0>
// CHECK-NEXT:  	vzeroupper
// CHECK-NEXT:  	ret
