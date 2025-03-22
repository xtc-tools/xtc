// RUN: mlir-loop %s --no-alias --always-vectorize --arch x86-64 --cpu tigerlake --print-assembly --hide-jumps 2>&1 | filecheck %s

func.func @myfun(
  %A: memref<512x1024xf32>,
  %B: memref<1024x128xf32>,
  %C: memref<512x128xf32>
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
    outs(%C : memref<512x128xf32>)
  linalg.matmul
    {
      loop.dims = ["i","j","k"],
      loop.tiles_names = {"i" = ["i1"], "j" = ["j1"], "k" = ["k1"]},
      loop.tiles_sizes = {i1 = 4, j1 = 64, k1 = 8},
      loop.interchange = ["i","j","k","k1","i1","j1"],
      loop.vectorize = ["j1"],
      loop.unroll = {i1 = 4, k1 = 8}
    }
    ins(%A, %B : memref<512x1024xf32>, memref<1024x128xf32>)
    outs(%C : memref<512x128xf32>)
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
// CHECK-NEXT:  	add    $0xe00,%r14
// CHECK-NEXT:  	add    $0x301c,%r15
// CHECK-NEXT:  	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
// CHECK-NEXT:  	mov    %r12,%rax
// CHECK-NEXT:  	shl    $0x9,%rax
// CHECK-NEXT:  	add    %rbx,%rax
// CHECK-NEXT:  	mov    $0x1,%cl
// CHECK-NEXT:  	xor    %edx,%edx
// CHECK-NEXT:  	xchg   %ax,%ax
// CHECK-NEXT:  	vmovups (%rax,%rdx,4),%zmm6
// CHECK-NEXT:  	vmovups 0x40(%rax,%rdx,4),%zmm7
// CHECK-NEXT:  	vmovups 0x80(%rax,%rdx,4),%zmm8
// CHECK-NEXT:  	vmovups 0xc0(%rax,%rdx,4),%zmm9
// CHECK-NEXT:  	vmovups 0x2c0(%rax,%rdx,4),%zmm0
// CHECK-NEXT:  	vmovups 0x280(%rax,%rdx,4),%zmm3
// CHECK-NEXT:  	vmovups 0x240(%rax,%rdx,4),%zmm10
// CHECK-NEXT:  	vmovups 0x200(%rax,%rdx,4),%zmm13
// CHECK-NEXT:  	vmovups 0x4c0(%rax,%rdx,4),%zmm1
// CHECK-NEXT:  	vmovups 0x480(%rax,%rdx,4),%zmm4
// CHECK-NEXT:  	vmovups 0x440(%rax,%rdx,4),%zmm11
// CHECK-NEXT:  	vmovups 0x400(%rax,%rdx,4),%zmm14
// CHECK-NEXT:  	vmovups 0x6c0(%rax,%rdx,4),%zmm2
// CHECK-NEXT:  	vmovups 0x680(%rax,%rdx,4),%zmm5
// CHECK-NEXT:  	vmovups 0x640(%rax,%rdx,4),%zmm12
// CHECK-NEXT:  	vmovups 0x600(%rax,%rdx,4),%zmm15
// CHECK-NEXT:  	lea    (%r14,%rdx,4),%rsi
// CHECK-NEXT:  	mov    $0xfffffffffffffff8,%rdi
// CHECK-NEXT:  	nopw   0x0(%rax,%rax,1)
// CHECK-NEXT:  	vmovups -0xe00(%rsi),%zmm16
// CHECK-NEXT:  	vmovups -0xdc0(%rsi),%zmm17
// CHECK-NEXT:  	vmovups -0xd80(%rsi),%zmm18
// CHECK-NEXT:  	vmovups -0xd40(%rsi),%zmm19
// CHECK-NEXT:  	vbroadcastss -0x2ffc(%r15,%rdi,4),%zmm20
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm20,%zmm9
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm20,%zmm8
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm20,%zmm6
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm17,%zmm7
// CHECK-NEXT:  	vbroadcastss -0x1ffc(%r15,%rdi,4),%zmm20
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm20,%zmm0
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm20,%zmm3
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm20,%zmm13
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm17,%zmm10
// CHECK-NEXT:  	vbroadcastss -0xffc(%r15,%rdi,4),%zmm20
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm20,%zmm1
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm20,%zmm4
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm20,%zmm14
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm17,%zmm11
// CHECK-NEXT:  	vbroadcastss 0x4(%r15,%rdi,4),%zmm20
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm20,%zmm2
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm20,%zmm5
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm20,%zmm15
// CHECK-NEXT:  	vmovups -0xb40(%rsi),%zmm16
// CHECK-NEXT:  	vmovups -0xb80(%rsi),%zmm18
// CHECK-NEXT:  	vmovups -0xc00(%rsi),%zmm19
// CHECK-NEXT:  	vfmadd231ps %zmm17,%zmm20,%zmm12
// CHECK-NEXT:  	vmovups -0xbc0(%rsi),%zmm17
// CHECK-NEXT:  	vbroadcastss -0x2ff8(%r15,%rdi,4),%zmm20
// CHECK-NEXT:  	vfmadd231ps %zmm17,%zmm20,%zmm7
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm20,%zmm6
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm20,%zmm8
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm16,%zmm9
// CHECK-NEXT:  	vbroadcastss -0x1ff8(%r15,%rdi,4),%zmm20
// CHECK-NEXT:  	vfmadd231ps %zmm17,%zmm20,%zmm10
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm20,%zmm13
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm20,%zmm3
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm16,%zmm0
// CHECK-NEXT:  	vbroadcastss -0xff8(%r15,%rdi,4),%zmm20
// CHECK-NEXT:  	vfmadd231ps %zmm17,%zmm20,%zmm11
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm20,%zmm14
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm20,%zmm4
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm16,%zmm1
// CHECK-NEXT:  	vbroadcastss 0x8(%r15,%rdi,4),%zmm20
// CHECK-NEXT:  	vfmadd231ps %zmm17,%zmm20,%zmm12
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm20,%zmm15
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm20,%zmm5
// CHECK-NEXT:  	vmovups -0x9c0(%rsi),%zmm17
// CHECK-NEXT:  	vmovups -0xa00(%rsi),%zmm18
// CHECK-NEXT:  	vmovups -0x980(%rsi),%zmm19
// CHECK-NEXT:  	vmovups -0x940(%rsi),%zmm21
// CHECK-NEXT:  	vbroadcastss -0x2ff4(%r15,%rdi,4),%zmm22
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm20,%zmm2
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm22,%zmm9
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm22,%zmm8
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm22,%zmm6
// CHECK-NEXT:  	vbroadcastss -0x1ff4(%r15,%rdi,4),%zmm16
// CHECK-NEXT:  	vfmadd231ps %zmm22,%zmm17,%zmm7
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm16,%zmm0
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm16,%zmm3
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm16,%zmm13
// CHECK-NEXT:  	vbroadcastss -0xff4(%r15,%rdi,4),%zmm20
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm17,%zmm10
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm20,%zmm1
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm20,%zmm4
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm20,%zmm14
// CHECK-NEXT:  	vbroadcastss 0xc(%r15,%rdi,4),%zmm16
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm17,%zmm11
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm16,%zmm2
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm16,%zmm5
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm16,%zmm15
// CHECK-NEXT:  	vmovups -0x740(%rsi),%zmm18
// CHECK-NEXT:  	vmovups -0x780(%rsi),%zmm19
// CHECK-NEXT:  	vmovups -0x800(%rsi),%zmm20
// CHECK-NEXT:  	vmovups -0x7c0(%rsi),%zmm21
// CHECK-NEXT:  	vfmadd231ps %zmm17,%zmm16,%zmm12
// CHECK-NEXT:  	vbroadcastss -0x2ff0(%r15,%rdi,4),%zmm16
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm16,%zmm7
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm16,%zmm6
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm16,%zmm8
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm18,%zmm9
// CHECK-NEXT:  	vbroadcastss -0x1ff0(%r15,%rdi,4),%zmm16
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm16,%zmm10
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm16,%zmm13
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm16,%zmm3
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm18,%zmm0
// CHECK-NEXT:  	vbroadcastss -0xff0(%r15,%rdi,4),%zmm16
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm16,%zmm11
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm16,%zmm14
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm16,%zmm4
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm18,%zmm1
// CHECK-NEXT:  	vbroadcastss 0x10(%r15,%rdi,4),%zmm16
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm16,%zmm12
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm16,%zmm15
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm16,%zmm5
// CHECK-NEXT:  	vmovups -0x5c0(%rsi),%zmm17
// CHECK-NEXT:  	vmovups -0x600(%rsi),%zmm19
// CHECK-NEXT:  	vmovups -0x580(%rsi),%zmm20
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm16,%zmm2
// CHECK-NEXT:  	vmovups -0x540(%rsi),%zmm16
// CHECK-NEXT:  	vbroadcastss -0x2fec(%r15,%rdi,4),%zmm18
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm18,%zmm9
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm18,%zmm8
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm18,%zmm6
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm17,%zmm7
// CHECK-NEXT:  	vbroadcastss -0x1fec(%r15,%rdi,4),%zmm18
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm18,%zmm0
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm18,%zmm3
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm18,%zmm13
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm17,%zmm10
// CHECK-NEXT:  	vbroadcastss -0xfec(%r15,%rdi,4),%zmm18
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm18,%zmm1
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm18,%zmm4
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm18,%zmm14
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm17,%zmm11
// CHECK-NEXT:  	vbroadcastss 0x14(%r15,%rdi,4),%zmm18
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm18,%zmm2
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm18,%zmm5
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm18,%zmm15
// CHECK-NEXT:  	vmovups -0x340(%rsi),%zmm16
// CHECK-NEXT:  	vmovups -0x380(%rsi),%zmm19
// CHECK-NEXT:  	vmovups -0x400(%rsi),%zmm20
// CHECK-NEXT:  	vmovups -0x3c0(%rsi),%zmm21
// CHECK-NEXT:  	vbroadcastss -0x2fe8(%r15,%rdi,4),%zmm22
// CHECK-NEXT:  	vfmadd231ps %zmm17,%zmm18,%zmm12
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm22,%zmm7
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm22,%zmm6
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm22,%zmm8
// CHECK-NEXT:  	vbroadcastss -0x1fe8(%r15,%rdi,4),%zmm17
// CHECK-NEXT:  	vfmadd231ps %zmm22,%zmm16,%zmm9
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm17,%zmm10
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm17,%zmm13
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm17,%zmm3
// CHECK-NEXT:  	vbroadcastss -0xfe8(%r15,%rdi,4),%zmm18
// CHECK-NEXT:  	vfmadd231ps %zmm17,%zmm16,%zmm0
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm18,%zmm11
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm18,%zmm14
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm18,%zmm4
// CHECK-NEXT:  	vbroadcastss 0x18(%r15,%rdi,4),%zmm17
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm16,%zmm1
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm17,%zmm12
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm17,%zmm15
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm17,%zmm5
// CHECK-NEXT:  	vmovups -0x1c0(%rsi),%zmm18
// CHECK-NEXT:  	vmovups -0x200(%rsi),%zmm19
// CHECK-NEXT:  	vmovups -0x180(%rsi),%zmm20
// CHECK-NEXT:  	vmovups -0x140(%rsi),%zmm21
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm17,%zmm2
// CHECK-NEXT:  	vbroadcastss -0x2fe4(%r15,%rdi,4),%zmm16
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm16,%zmm9
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm16,%zmm8
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm16,%zmm6
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm18,%zmm7
// CHECK-NEXT:  	vbroadcastss -0x1fe4(%r15,%rdi,4),%zmm16
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm16,%zmm0
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm16,%zmm3
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm16,%zmm13
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm18,%zmm10
// CHECK-NEXT:  	vbroadcastss -0xfe4(%r15,%rdi,4),%zmm16
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm16,%zmm1
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm16,%zmm4
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm16,%zmm14
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm18,%zmm11
// CHECK-NEXT:  	vbroadcastss 0x1c(%r15,%rdi,4),%zmm16
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm16,%zmm2
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm16,%zmm5
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm16,%zmm15
// CHECK-NEXT:  	vmovups 0xc0(%rsi),%zmm17
// CHECK-NEXT:  	vmovups 0x80(%rsi),%zmm19
// CHECK-NEXT:  	vmovups (%rsi),%zmm20
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm16,%zmm12
// CHECK-NEXT:  	vmovups 0x40(%rsi),%zmm16
// CHECK-NEXT:  	vbroadcastss -0x2fe0(%r15,%rdi,4),%zmm18
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm18,%zmm7
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm18,%zmm6
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm18,%zmm8
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm17,%zmm9
// CHECK-NEXT:  	vbroadcastss -0x1fe0(%r15,%rdi,4),%zmm18
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm18,%zmm10
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm18,%zmm13
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm18,%zmm3
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm17,%zmm0
// CHECK-NEXT:  	vbroadcastss -0xfe0(%r15,%rdi,4),%zmm18
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm18,%zmm11
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm18,%zmm14
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm18,%zmm4
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm17,%zmm1
// CHECK-NEXT:  	vbroadcastss 0x20(%r15,%rdi,4),%zmm18
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm18,%zmm12
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm18,%zmm15
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm18,%zmm5
// CHECK-NEXT:  	vfmadd231ps %zmm17,%zmm18,%zmm2
// CHECK-NEXT:  	add    $0x8,%rdi
// CHECK-NEXT:  	add    $0x1000,%rsi
// CHECK-NEXT:  	cmp    $0x3f8,%rdi
// CHECK-NEXT:  	jb     <myfun+0xe0>
// CHECK-NEXT:  	vmovups %zmm6,(%rax,%rdx,4)
// CHECK-NEXT:  	vmovups %zmm7,0x40(%rax,%rdx,4)
// CHECK-NEXT:  	vmovups %zmm8,0x80(%rax,%rdx,4)
// CHECK-NEXT:  	vmovups %zmm9,0xc0(%rax,%rdx,4)
// CHECK-NEXT:  	vmovups %zmm13,0x200(%rax,%rdx,4)
// CHECK-NEXT:  	vmovups %zmm10,0x240(%rax,%rdx,4)
// CHECK-NEXT:  	vmovups %zmm3,0x280(%rax,%rdx,4)
// CHECK-NEXT:  	vmovups %zmm0,0x2c0(%rax,%rdx,4)
// CHECK-NEXT:  	vmovups %zmm14,0x400(%rax,%rdx,4)
// CHECK-NEXT:  	vmovups %zmm11,0x440(%rax,%rdx,4)
// CHECK-NEXT:  	vmovups %zmm4,0x480(%rax,%rdx,4)
// CHECK-NEXT:  	vmovups %zmm1,0x4c0(%rax,%rdx,4)
// CHECK-NEXT:  	vmovups %zmm15,0x600(%rax,%rdx,4)
// CHECK-NEXT:  	vmovups %zmm12,0x640(%rax,%rdx,4)
// CHECK-NEXT:  	vmovups %zmm5,0x680(%rax,%rdx,4)
// CHECK-NEXT:  	vmovups %zmm2,0x6c0(%rax,%rdx,4)
// CHECK-NEXT:  	mov    $0x40,%edx
// CHECK-NEXT:  	test   $0x1,%cl
// CHECK-NEXT:  	mov    $0x0,%ecx
// CHECK-NEXT:  	jne    <myfun+0x50>
// CHECK-NEXT:  	lea    0x4(%r12),%rax
// CHECK-NEXT:  	add    $0x4000,%r15
// CHECK-NEXT:  	cmp    $0x1fc,%r12
// CHECK-NEXT:  	mov    %rax,%r12
// CHECK-NEXT:  	jb     <myfun+0x40>
// CHECK-NEXT:  	add    $0x8,%rsp
// CHECK-NEXT:  	pop    %rbx
// CHECK-NEXT:  	pop    %r12
// CHECK-NEXT:  	pop    %r14
// CHECK-NEXT:  	pop    %r15
// CHECK-NEXT:  	vzeroupper
// CHECK-NEXT:  	ret
