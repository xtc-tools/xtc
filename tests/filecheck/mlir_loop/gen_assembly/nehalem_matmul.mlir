// RUN: mlir-loop --no-alias --arch x86-64 --cpu nehalem --print-assembly --hide-jumps %s 2>&1 | filecheck %s

func.func @myfun(
  %A: memref<256x512xf32>,
  %B: memref<512x256xf32>,
  %C: memref<256x256xf32>
) {
  linalg.matmul
    {
      loop.dims = ["I","J","K"],
      loop.schedule = {
        "I",
          "J",
            "K",
              "I#1" = {"unroll"},
                "K#4"= {"unroll"},
                  "J#16" = {"vectorize"}
      }
    }
    ins(%A, %B : memref<256x512xf32>, memref<512x256xf32>)
    outs(%C : memref<256x256xf32>)
  return
}
// CHECK:       <myfun>:
// CHECK-NEXT:  	add    $0xc00,%rsi
// CHECK-NEXT:  	add    $0xc,%rdi
// CHECK-NEXT:  	xor    %eax,%eax
// CHECK-NEXT:  	nopl   (%rax)
// CHECK-NEXT:  	mov    %rax,%rcx
// CHECK-NEXT:  	shl    $0xa,%rcx
// CHECK-NEXT:  	add    %rdx,%rcx
// CHECK-NEXT:  	mov    %rsi,%r8
// CHECK-NEXT:  	xor    %r9d,%r9d
// CHECK-NEXT:  	movups (%rcx,%r9,4),%xmm1
// CHECK-NEXT:  	movups 0x10(%rcx,%r9,4),%xmm3
// CHECK-NEXT:  	movups 0x20(%rcx,%r9,4),%xmm4
// CHECK-NEXT:  	movups 0x30(%rcx,%r9,4),%xmm0
// CHECK-NEXT:  	mov    $0xfffffffffffffffc,%r10
// CHECK-NEXT:  	mov    %r8,%r11
// CHECK-NEXT:  	cs nopw 0x0(%rax,%rax,1)
// CHECK-NEXT:  	nopl   0x0(%rax,%rax,1)
// CHECK-NEXT:  	movups -0xc00(%r11),%xmm9
// CHECK-NEXT:  	movups -0xbf0(%r11),%xmm8
// CHECK-NEXT:  	movups -0xbe0(%r11),%xmm7
// CHECK-NEXT:  	movups -0xbd0(%r11),%xmm6
// CHECK-NEXT:  	movss  0x4(%rdi,%r10,4),%xmm10
// CHECK-NEXT:  	movss  0x8(%rdi,%r10,4),%xmm5
// CHECK-NEXT:  	movss  0xc(%rdi,%r10,4),%xmm2
// CHECK-NEXT:  	shufps $0x0,%xmm10,%xmm10
// CHECK-NEXT:  	mulps  %xmm10,%xmm6
// CHECK-NEXT:  	addps  %xmm0,%xmm6
// CHECK-NEXT:  	movss  0x10(%rdi,%r10,4),%xmm0
// CHECK-NEXT:  	mulps  %xmm10,%xmm7
// CHECK-NEXT:  	addps  %xmm4,%xmm7
// CHECK-NEXT:  	mulps  %xmm10,%xmm8
// CHECK-NEXT:  	addps  %xmm3,%xmm8
// CHECK-NEXT:  	mulps  %xmm9,%xmm10
// CHECK-NEXT:  	addps  %xmm1,%xmm10
// CHECK-NEXT:  	movups -0x7d0(%r11),%xmm9
// CHECK-NEXT:  	movups -0x7e0(%r11),%xmm4
// CHECK-NEXT:  	movups -0x7f0(%r11),%xmm3
// CHECK-NEXT:  	movups -0x800(%r11),%xmm1
// CHECK-NEXT:  	shufps $0x0,%xmm5,%xmm5
// CHECK-NEXT:  	mulps  %xmm5,%xmm1
// CHECK-NEXT:  	addps  %xmm10,%xmm1
// CHECK-NEXT:  	mulps  %xmm5,%xmm3
// CHECK-NEXT:  	addps  %xmm8,%xmm3
// CHECK-NEXT:  	mulps  %xmm5,%xmm4
// CHECK-NEXT:  	addps  %xmm7,%xmm4
// CHECK-NEXT:  	mulps  %xmm9,%xmm5
// CHECK-NEXT:  	addps  %xmm6,%xmm5
// CHECK-NEXT:  	movups -0x400(%r11),%xmm9
// CHECK-NEXT:  	movups -0x3f0(%r11),%xmm8
// CHECK-NEXT:  	movups -0x3e0(%r11),%xmm7
// CHECK-NEXT:  	movups -0x3d0(%r11),%xmm6
// CHECK-NEXT:  	shufps $0x0,%xmm2,%xmm2
// CHECK-NEXT:  	mulps  %xmm2,%xmm6
// CHECK-NEXT:  	addps  %xmm5,%xmm6
// CHECK-NEXT:  	mulps  %xmm2,%xmm7
// CHECK-NEXT:  	addps  %xmm4,%xmm7
// CHECK-NEXT:  	mulps  %xmm2,%xmm8
// CHECK-NEXT:  	addps  %xmm3,%xmm8
// CHECK-NEXT:  	mulps  %xmm9,%xmm2
// CHECK-NEXT:  	addps  %xmm1,%xmm2
// CHECK-NEXT:  	movups 0x30(%r11),%xmm5
// CHECK-NEXT:  	movups 0x20(%r11),%xmm4
// CHECK-NEXT:  	movups 0x10(%r11),%xmm3
// CHECK-NEXT:  	movups (%r11),%xmm1
// CHECK-NEXT:  	shufps $0x0,%xmm0,%xmm0
// CHECK-NEXT:  	mulps  %xmm0,%xmm1
// CHECK-NEXT:  	addps  %xmm2,%xmm1
// CHECK-NEXT:  	mulps  %xmm0,%xmm3
// CHECK-NEXT:  	addps  %xmm8,%xmm3
// CHECK-NEXT:  	mulps  %xmm0,%xmm4
// CHECK-NEXT:  	addps  %xmm7,%xmm4
// CHECK-NEXT:  	mulps  %xmm5,%xmm0
// CHECK-NEXT:  	addps  %xmm6,%xmm0
// CHECK-NEXT:  	add    $0x4,%r10
// CHECK-NEXT:  	add    $0x1000,%r11
// CHECK-NEXT:  	cmp    $0x1fc,%r10
// CHECK-NEXT:  	jb     <myfun+0x50>
// CHECK-NEXT:  	movups %xmm1,(%rcx,%r9,4)
// CHECK-NEXT:  	movups %xmm3,0x10(%rcx,%r9,4)
// CHECK-NEXT:  	movups %xmm4,0x20(%rcx,%r9,4)
// CHECK-NEXT:  	movups %xmm0,0x30(%rcx,%r9,4)
// CHECK-NEXT:  	lea    0x10(%r9),%r10
// CHECK-NEXT:  	add    $0x40,%r8
// CHECK-NEXT:  	cmp    $0xf0,%r9
// CHECK-NEXT:  	mov    %r10,%r9
// CHECK-NEXT:  	jb     <myfun+0x20>
// CHECK-NEXT:  	lea    0x1(%rax),%rcx
// CHECK-NEXT:  	add    $0x800,%rdi
// CHECK-NEXT:  	cmp    $0xff,%rax
// CHECK-NEXT:  	mov    %rcx,%rax
// CHECK-NEXT:  	jb     <myfun+0x10>
// CHECK-NEXT:  	ret
