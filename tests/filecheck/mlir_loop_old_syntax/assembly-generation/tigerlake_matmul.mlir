// RUN: mlir-loop %s --no-alias --arch x86-64 --cpu tigerlake --print-assembly --hide-jumps 2>&1 | filecheck %s

func.func @myfun(
  %A: memref<512x1024xf32>,
  %B: memref<1024x128xf32>,
  %C: memref<512x128xf32>
) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.fill
       {
        loop.dims = ["i","j"],
        loop.tiles = {"i" = {"i1" = 4}, "j" = {"j1" = 64}},
        loop.interchange = ["i","j","i1","j1"],
        loop.vectorize = ["j1"]
    }
    ins(%cst : f32)
    outs(%C : memref<512x128xf32>)
  linalg.matmul
    {
      loop.dims = ["i","j","k"],
      loop.tiles = {"i" = {"i1" = 4}, "j" = {"j1" = 64}, "k" = {"k1" = 8}},
      loop.interchange = ["i","j","k","k1","i1","j1"],
      loop.vectorize = ["j1"],
      loop.unroll = {i1 = 4, k1 = 8}
    }
    ins(%A, %B : memref<512x1024xf32>, memref<1024x128xf32>)
    outs(%C : memref<512x128xf32>)
  return
}
// CHECK:       <myfun>:
// CHECK-NEXT:  	lea    0x700(%rdx),%rax
// CHECK-NEXT:  	mov    $0xfffffffffffffffc,%rcx
// CHECK-NEXT:  	vxorps %xmm0,%xmm0,%xmm0
// CHECK-NEXT:  	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
// CHECK-NEXT:  	vmovups %zmm0,-0x640(%rax)
// CHECK-NEXT:  	vmovups %zmm0,-0x680(%rax)
// CHECK-NEXT:  	vmovups %zmm0,-0x6c0(%rax)
// CHECK-NEXT:  	vmovups %zmm0,-0x700(%rax)
// CHECK-NEXT:  	vmovups %zmm0,-0x440(%rax)
// CHECK-NEXT:  	vmovups %zmm0,-0x480(%rax)
// CHECK-NEXT:  	vmovups %zmm0,-0x4c0(%rax)
// CHECK-NEXT:  	vmovups %zmm0,-0x500(%rax)
// CHECK-NEXT:  	vmovups %zmm0,-0x240(%rax)
// CHECK-NEXT:  	vmovups %zmm0,-0x280(%rax)
// CHECK-NEXT:  	vmovups %zmm0,-0x2c0(%rax)
// CHECK-NEXT:  	vmovups %zmm0,-0x300(%rax)
// CHECK-NEXT:  	vmovups %zmm0,-0x40(%rax)
// CHECK-NEXT:  	vmovups %zmm0,-0x80(%rax)
// CHECK-NEXT:  	vmovups %zmm0,-0xc0(%rax)
// CHECK-NEXT:  	vmovups %zmm0,-0x100(%rax)
// CHECK-NEXT:  	vmovups %zmm0,-0x540(%rax)
// CHECK-NEXT:  	vmovups %zmm0,-0x580(%rax)
// CHECK-NEXT:  	vmovups %zmm0,-0x5c0(%rax)
// CHECK-NEXT:  	vmovups %zmm0,-0x600(%rax)
// CHECK-NEXT:  	vmovups %zmm0,-0x340(%rax)
// CHECK-NEXT:  	vmovups %zmm0,-0x380(%rax)
// CHECK-NEXT:  	vmovups %zmm0,-0x3c0(%rax)
// CHECK-NEXT:  	vmovups %zmm0,-0x400(%rax)
// CHECK-NEXT:  	vmovups %zmm0,-0x140(%rax)
// CHECK-NEXT:  	vmovups %zmm0,-0x180(%rax)
// CHECK-NEXT:  	vmovups %zmm0,-0x1c0(%rax)
// CHECK-NEXT:  	vmovups %zmm0,-0x200(%rax)
// CHECK-NEXT:  	vmovups %zmm0,0xc0(%rax)
// CHECK-NEXT:  	vmovups %zmm0,0x80(%rax)
// CHECK-NEXT:  	vmovups %zmm0,0x40(%rax)
// CHECK-NEXT:  	vmovups %zmm0,(%rax)
// CHECK-NEXT:  	add    $0x4,%rcx
// CHECK-NEXT:  	add    $0x800,%rax
// CHECK-NEXT:  	cmp    $0x1fc,%rcx
// CHECK-NEXT:  	jb     <myfun+0x20>
// CHECK-NEXT:  	add    $0xe00,%rsi
// CHECK-NEXT:  	add    $0x301c,%rdi
// CHECK-NEXT:  	xor    %eax,%eax
// CHECK-NEXT:  	cs nopw 0x0(%rax,%rax,1)
// CHECK-NEXT:  	mov    %rax,%rcx
// CHECK-NEXT:  	shl    $0x9,%rcx
// CHECK-NEXT:  	add    %rdx,%rcx
// CHECK-NEXT:  	mov    $0x1,%r8b
// CHECK-NEXT:  	xor    %r9d,%r9d
// CHECK-NEXT:  	vmovups (%rcx,%r9,4),%zmm6
// CHECK-NEXT:  	vmovups 0x40(%rcx,%r9,4),%zmm7
// CHECK-NEXT:  	vmovups 0x80(%rcx,%r9,4),%zmm8
// CHECK-NEXT:  	vmovups 0xc0(%rcx,%r9,4),%zmm9
// CHECK-NEXT:  	vmovups 0x2c0(%rcx,%r9,4),%zmm0
// CHECK-NEXT:  	vmovups 0x280(%rcx,%r9,4),%zmm3
// CHECK-NEXT:  	vmovups 0x240(%rcx,%r9,4),%zmm10
// CHECK-NEXT:  	vmovups 0x200(%rcx,%r9,4),%zmm13
// CHECK-NEXT:  	vmovups 0x4c0(%rcx,%r9,4),%zmm1
// CHECK-NEXT:  	vmovups 0x480(%rcx,%r9,4),%zmm4
// CHECK-NEXT:  	vmovups 0x440(%rcx,%r9,4),%zmm11
// CHECK-NEXT:  	vmovups 0x400(%rcx,%r9,4),%zmm14
// CHECK-NEXT:  	vmovups 0x6c0(%rcx,%r9,4),%zmm2
// CHECK-NEXT:  	vmovups 0x680(%rcx,%r9,4),%zmm5
// CHECK-NEXT:  	vmovups 0x640(%rcx,%r9,4),%zmm12
// CHECK-NEXT:  	vmovups 0x600(%rcx,%r9,4),%zmm15
// CHECK-NEXT:  	lea    (%rsi,%r9,4),%r10
// CHECK-NEXT:  	mov    $0xfffffffffffffff8,%r11
// CHECK-NEXT:  	nopw   0x0(%rax,%rax,1)
// CHECK-NEXT:  	vmovups -0xe00(%r10),%zmm16
// CHECK-NEXT:  	vmovups -0xdc0(%r10),%zmm17
// CHECK-NEXT:  	vmovups -0xd80(%r10),%zmm18
// CHECK-NEXT:  	vmovups -0xd40(%r10),%zmm19
// CHECK-NEXT:  	vbroadcastss -0x2ffc(%rdi,%r11,4),%zmm20
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm20,%zmm9
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm20,%zmm8
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm20,%zmm6
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm17,%zmm7
// CHECK-NEXT:  	vbroadcastss -0x1ffc(%rdi,%r11,4),%zmm20
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm20,%zmm0
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm20,%zmm3
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm20,%zmm13
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm17,%zmm10
// CHECK-NEXT:  	vbroadcastss -0xffc(%rdi,%r11,4),%zmm20
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm20,%zmm1
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm20,%zmm4
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm20,%zmm14
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm17,%zmm11
// CHECK-NEXT:  	vbroadcastss 0x4(%rdi,%r11,4),%zmm20
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm20,%zmm2
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm20,%zmm5
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm20,%zmm15
// CHECK-NEXT:  	vmovups -0xb40(%r10),%zmm16
// CHECK-NEXT:  	vmovups -0xb80(%r10),%zmm18
// CHECK-NEXT:  	vmovups -0xc00(%r10),%zmm19
// CHECK-NEXT:  	vfmadd231ps %zmm17,%zmm20,%zmm12
// CHECK-NEXT:  	vmovups -0xbc0(%r10),%zmm17
// CHECK-NEXT:  	vbroadcastss -0x2ff8(%rdi,%r11,4),%zmm20
// CHECK-NEXT:  	vfmadd231ps %zmm17,%zmm20,%zmm7
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm20,%zmm6
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm20,%zmm8
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm16,%zmm9
// CHECK-NEXT:  	vbroadcastss -0x1ff8(%rdi,%r11,4),%zmm20
// CHECK-NEXT:  	vfmadd231ps %zmm17,%zmm20,%zmm10
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm20,%zmm13
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm20,%zmm3
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm16,%zmm0
// CHECK-NEXT:  	vbroadcastss -0xff8(%rdi,%r11,4),%zmm20
// CHECK-NEXT:  	vfmadd231ps %zmm17,%zmm20,%zmm11
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm20,%zmm14
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm20,%zmm4
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm16,%zmm1
// CHECK-NEXT:  	vbroadcastss 0x8(%rdi,%r11,4),%zmm20
// CHECK-NEXT:  	vfmadd231ps %zmm17,%zmm20,%zmm12
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm20,%zmm15
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm20,%zmm5
// CHECK-NEXT:  	vmovups -0x9c0(%r10),%zmm17
// CHECK-NEXT:  	vmovups -0xa00(%r10),%zmm18
// CHECK-NEXT:  	vmovups -0x980(%r10),%zmm19
// CHECK-NEXT:  	vmovups -0x940(%r10),%zmm21
// CHECK-NEXT:  	vbroadcastss -0x2ff4(%rdi,%r11,4),%zmm22
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm20,%zmm2
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm22,%zmm9
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm22,%zmm8
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm22,%zmm6
// CHECK-NEXT:  	vbroadcastss -0x1ff4(%rdi,%r11,4),%zmm16
// CHECK-NEXT:  	vfmadd231ps %zmm22,%zmm17,%zmm7
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm16,%zmm0
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm16,%zmm3
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm16,%zmm13
// CHECK-NEXT:  	vbroadcastss -0xff4(%rdi,%r11,4),%zmm20
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm17,%zmm10
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm20,%zmm1
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm20,%zmm4
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm20,%zmm14
// CHECK-NEXT:  	vbroadcastss 0xc(%rdi,%r11,4),%zmm16
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm17,%zmm11
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm16,%zmm2
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm16,%zmm5
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm16,%zmm15
// CHECK-NEXT:  	vmovups -0x740(%r10),%zmm18
// CHECK-NEXT:  	vmovups -0x780(%r10),%zmm19
// CHECK-NEXT:  	vmovups -0x800(%r10),%zmm20
// CHECK-NEXT:  	vmovups -0x7c0(%r10),%zmm21
// CHECK-NEXT:  	vfmadd231ps %zmm17,%zmm16,%zmm12
// CHECK-NEXT:  	vbroadcastss -0x2ff0(%rdi,%r11,4),%zmm16
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm16,%zmm7
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm16,%zmm6
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm16,%zmm8
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm18,%zmm9
// CHECK-NEXT:  	vbroadcastss -0x1ff0(%rdi,%r11,4),%zmm16
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm16,%zmm10
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm16,%zmm13
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm16,%zmm3
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm18,%zmm0
// CHECK-NEXT:  	vbroadcastss -0xff0(%rdi,%r11,4),%zmm16
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm16,%zmm11
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm16,%zmm14
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm16,%zmm4
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm18,%zmm1
// CHECK-NEXT:  	vbroadcastss 0x10(%rdi,%r11,4),%zmm16
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm16,%zmm12
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm16,%zmm15
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm16,%zmm5
// CHECK-NEXT:  	vmovups -0x5c0(%r10),%zmm17
// CHECK-NEXT:  	vmovups -0x600(%r10),%zmm19
// CHECK-NEXT:  	vmovups -0x580(%r10),%zmm20
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm16,%zmm2
// CHECK-NEXT:  	vmovups -0x540(%r10),%zmm16
// CHECK-NEXT:  	vbroadcastss -0x2fec(%rdi,%r11,4),%zmm18
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm18,%zmm9
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm18,%zmm8
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm18,%zmm6
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm17,%zmm7
// CHECK-NEXT:  	vbroadcastss -0x1fec(%rdi,%r11,4),%zmm18
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm18,%zmm0
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm18,%zmm3
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm18,%zmm13
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm17,%zmm10
// CHECK-NEXT:  	vbroadcastss -0xfec(%rdi,%r11,4),%zmm18
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm18,%zmm1
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm18,%zmm4
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm18,%zmm14
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm17,%zmm11
// CHECK-NEXT:  	vbroadcastss 0x14(%rdi,%r11,4),%zmm18
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm18,%zmm2
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm18,%zmm5
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm18,%zmm15
// CHECK-NEXT:  	vmovups -0x340(%r10),%zmm16
// CHECK-NEXT:  	vmovups -0x380(%r10),%zmm19
// CHECK-NEXT:  	vmovups -0x400(%r10),%zmm20
// CHECK-NEXT:  	vmovups -0x3c0(%r10),%zmm21
// CHECK-NEXT:  	vbroadcastss -0x2fe8(%rdi,%r11,4),%zmm22
// CHECK-NEXT:  	vfmadd231ps %zmm17,%zmm18,%zmm12
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm22,%zmm7
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm22,%zmm6
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm22,%zmm8
// CHECK-NEXT:  	vbroadcastss -0x1fe8(%rdi,%r11,4),%zmm17
// CHECK-NEXT:  	vfmadd231ps %zmm22,%zmm16,%zmm9
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm17,%zmm10
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm17,%zmm13
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm17,%zmm3
// CHECK-NEXT:  	vbroadcastss -0xfe8(%rdi,%r11,4),%zmm18
// CHECK-NEXT:  	vfmadd231ps %zmm17,%zmm16,%zmm0
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm18,%zmm11
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm18,%zmm14
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm18,%zmm4
// CHECK-NEXT:  	vbroadcastss 0x18(%rdi,%r11,4),%zmm17
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm16,%zmm1
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm17,%zmm12
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm17,%zmm15
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm17,%zmm5
// CHECK-NEXT:  	vmovups -0x1c0(%r10),%zmm18
// CHECK-NEXT:  	vmovups -0x200(%r10),%zmm19
// CHECK-NEXT:  	vmovups -0x180(%r10),%zmm20
// CHECK-NEXT:  	vmovups -0x140(%r10),%zmm21
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm17,%zmm2
// CHECK-NEXT:  	vbroadcastss -0x2fe4(%rdi,%r11,4),%zmm16
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm16,%zmm9
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm16,%zmm8
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm16,%zmm6
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm18,%zmm7
// CHECK-NEXT:  	vbroadcastss -0x1fe4(%rdi,%r11,4),%zmm16
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm16,%zmm0
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm16,%zmm3
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm16,%zmm13
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm18,%zmm10
// CHECK-NEXT:  	vbroadcastss -0xfe4(%rdi,%r11,4),%zmm16
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm16,%zmm1
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm16,%zmm4
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm16,%zmm14
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm18,%zmm11
// CHECK-NEXT:  	vbroadcastss 0x1c(%rdi,%r11,4),%zmm16
// CHECK-NEXT:  	vfmadd231ps %zmm21,%zmm16,%zmm2
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm16,%zmm5
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm16,%zmm15
// CHECK-NEXT:  	vmovups 0xc0(%r10),%zmm17
// CHECK-NEXT:  	vmovups 0x80(%r10),%zmm19
// CHECK-NEXT:  	vmovups (%r10),%zmm20
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm16,%zmm12
// CHECK-NEXT:  	vmovups 0x40(%r10),%zmm16
// CHECK-NEXT:  	vbroadcastss -0x2fe0(%rdi,%r11,4),%zmm18
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm18,%zmm7
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm18,%zmm6
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm18,%zmm8
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm17,%zmm9
// CHECK-NEXT:  	vbroadcastss -0x1fe0(%rdi,%r11,4),%zmm18
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm18,%zmm10
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm18,%zmm13
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm18,%zmm3
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm17,%zmm0
// CHECK-NEXT:  	vbroadcastss -0xfe0(%rdi,%r11,4),%zmm18
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm18,%zmm11
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm18,%zmm14
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm18,%zmm4
// CHECK-NEXT:  	vfmadd231ps %zmm18,%zmm17,%zmm1
// CHECK-NEXT:  	vbroadcastss 0x20(%rdi,%r11,4),%zmm18
// CHECK-NEXT:  	vfmadd231ps %zmm16,%zmm18,%zmm12
// CHECK-NEXT:  	vfmadd231ps %zmm20,%zmm18,%zmm15
// CHECK-NEXT:  	vfmadd231ps %zmm19,%zmm18,%zmm5
// CHECK-NEXT:  	vfmadd231ps %zmm17,%zmm18,%zmm2
// CHECK-NEXT:  	add    $0x8,%r11
// CHECK-NEXT:  	add    $0x1000,%r10
// CHECK-NEXT:  	cmp    $0x3f8,%r11
// CHECK-NEXT:  	jb     <myfun+0x1d0>
// CHECK-NEXT:  	vmovups %zmm6,(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %zmm7,0x40(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %zmm8,0x80(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %zmm9,0xc0(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %zmm13,0x200(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %zmm10,0x240(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %zmm3,0x280(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %zmm0,0x2c0(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %zmm14,0x400(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %zmm11,0x440(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %zmm4,0x480(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %zmm1,0x4c0(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %zmm15,0x600(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %zmm12,0x640(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %zmm5,0x680(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %zmm2,0x6c0(%rcx,%r9,4)
// CHECK-NEXT:  	mov    $0x40,%r9d
// CHECK-NEXT:  	test   $0x1,%r8b
// CHECK-NEXT:  	mov    $0x0,%r8d
// CHECK-NEXT:  	jne    <myfun+0x140>
// CHECK-NEXT:  	lea    0x4(%rax),%rcx
// CHECK-NEXT:  	add    $0x4000,%rdi
// CHECK-NEXT:  	cmp    $0x1fc,%rax
// CHECK-NEXT:  	mov    %rcx,%rax
// CHECK-NEXT:  	jb     <myfun+0x130>
// CHECK-NEXT:  	vzeroupper
// CHECK-NEXT:  	ret
