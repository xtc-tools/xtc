// RUN: mlir-loop --no-alias --arch x86-64 --cpu skylake --print-assembly --hide-jumps %s 2>&1 | filecheck %s
// UNSUPPORTED: mlir-target=c
// Assembly output will differ a bit when using C.

func.func @myfun(
  %A: memref<258x512xf32>,
  %B: memref<512x256xf32>,
  %C: memref<258x256xf32>
) {
  linalg.matmul
    {
      loop.dims = ["I","J","K"],
      loop.schedule = {
        "I[:2]" = {
          "I",
            "K",
              "J"
        },
        "I[2:]" = {
          "I",
            "J",
              "K",
                "I#1",
                  "K#8",
                    "J#64" = {"vectorize"}
        }
      }
    }
    ins(%A, %B : memref<258x512xf32>, memref<512x256xf32>)
    outs(%C : memref<258x256xf32>)
  return
}
// CHECK:       <myfun>:
// CHECK-NEXT:  	push   %rbp
// CHECK-NEXT:  	push   %r15
// CHECK-NEXT:  	push   %r14
// CHECK-NEXT:  	push   %r13
// CHECK-NEXT:  	push   %r12
// CHECK-NEXT:  	push   %rbx
// CHECK-NEXT:  	lea    0x100(%rdx),%rax
// CHECK-NEXT:  	mov    %rax,-0x8(%rsp)
// CHECK-NEXT:  	lea    0x200(%rdx),%rax
// CHECK-NEXT:  	mov    %rax,-0x10(%rsp)
// CHECK-NEXT:  	lea    0x300(%rdx),%rax
// CHECK-NEXT:  	mov    %rax,-0x18(%rsp)
// CHECK-NEXT:  	lea    0x1c00(%rsi),%rax
// CHECK-NEXT:  	mov    %rax,-0x20(%rsp)
// CHECK-NEXT:  	lea    0x1d00(%rsi),%rax
// CHECK-NEXT:  	mov    %rax,-0x28(%rsp)
// CHECK-NEXT:  	lea    0x1e00(%rsi),%rax
// CHECK-NEXT:  	mov    %rax,-0x30(%rsp)
// CHECK-NEXT:  	mov    %rsi,-0x38(%rsp)
// CHECK-NEXT:  	add    $0x1f00,%rsi
// CHECK-NEXT:  	mov    $0x1,%bpl
// CHECK-NEXT:  	xor    %r14d,%r14d
// CHECK-NEXT:  	data16 data16 cs nopw 0x0(%rax,%rax,1)
// CHECK-NEXT:  	mov    %r14,%r15
// CHECK-NEXT:  	shl    $0xb,%r15
// CHECK-NEXT:  	add    %rdi,%r15
// CHECK-NEXT:  	shl    $0xa,%r14
// CHECK-NEXT:  	mov    %rdx,%rcx
// CHECK-NEXT:  	lea    (%rdx,%r14,1),%r12
// CHECK-NEXT:  	mov    -0x8(%rsp),%rax
// CHECK-NEXT:  	lea    (%rax,%r14,1),%r13
// CHECK-NEXT:  	mov    -0x10(%rsp),%rax
// CHECK-NEXT:  	add    %r14,%rax
// CHECK-NEXT:  	add    -0x18(%rsp),%r14
// CHECK-NEXT:  	mov    %rsi,%rbx
// CHECK-NEXT:  	mov    -0x30(%rsp),%r11
// CHECK-NEXT:  	mov    -0x28(%rsp),%r10
// CHECK-NEXT:  	mov    -0x20(%rsp),%r9
// CHECK-NEXT:  	xor    %edx,%edx
// CHECK-NEXT:  	nop
// CHECK-NEXT:  	vmovss (%r15,%rdx,4),%xmm0
// CHECK-NEXT:  	vmovss 0x4(%r15,%rdx,4),%xmm1
// CHECK-NEXT:  	vmovss 0x8(%r15,%rdx,4),%xmm2
// CHECK-NEXT:  	vmovss 0xc(%r15,%rdx,4),%xmm3
// CHECK-NEXT:  	vmovss 0x10(%r15,%rdx,4),%xmm4
// CHECK-NEXT:  	vmovss 0x14(%r15,%rdx,4),%xmm5
// CHECK-NEXT:  	vmovss 0x18(%r15,%rdx,4),%xmm6
// CHECK-NEXT:  	vmovss 0x1c(%r15,%rdx,4),%xmm7
// CHECK-NEXT:  	mov    $0xffffffffffffffff,%r8
// CHECK-NEXT:  	xchg   %ax,%ax
// CHECK-NEXT:  	vmovss -0x1bfc(%r9,%r8,4),%xmm8
// CHECK-NEXT:  	vfmadd213ss 0x4(%r12,%r8,4),%xmm0,%xmm8
// CHECK-NEXT:  	vfmadd231ss -0x17fc(%r9,%r8,4),%xmm1,%xmm8
// CHECK-NEXT:  	vfmadd231ss -0x13fc(%r9,%r8,4),%xmm2,%xmm8
// CHECK-NEXT:  	vfmadd231ss -0xffc(%r9,%r8,4),%xmm3,%xmm8
// CHECK-NEXT:  	vfmadd231ss -0xbfc(%r9,%r8,4),%xmm4,%xmm8
// CHECK-NEXT:  	vfmadd231ss -0x7fc(%r9,%r8,4),%xmm5,%xmm8
// CHECK-NEXT:  	vfmadd231ss -0x3fc(%r9,%r8,4),%xmm6,%xmm8
// CHECK-NEXT:  	vfmadd231ss 0x4(%r9,%r8,4),%xmm7,%xmm8
// CHECK-NEXT:  	vmovss %xmm8,0x4(%r12,%r8,4)
// CHECK-NEXT:  	inc    %r8
// CHECK-NEXT:  	cmp    $0x3f,%r8
// CHECK-NEXT:  	jb     <myfun+0xf0>
// CHECK-NEXT:  	mov    $0xffffffffffffffff,%r8
// CHECK-NEXT:  	nopl   0x0(%rax,%rax,1)
// CHECK-NEXT:  	vmovss -0x1bfc(%r10,%r8,4),%xmm8
// CHECK-NEXT:  	vfmadd213ss 0x4(%r13,%r8,4),%xmm0,%xmm8
// CHECK-NEXT:  	vfmadd231ss -0x17fc(%r10,%r8,4),%xmm1,%xmm8
// CHECK-NEXT:  	vfmadd231ss -0x13fc(%r10,%r8,4),%xmm2,%xmm8
// CHECK-NEXT:  	vfmadd231ss -0xffc(%r10,%r8,4),%xmm3,%xmm8
// CHECK-NEXT:  	vfmadd231ss -0xbfc(%r10,%r8,4),%xmm4,%xmm8
// CHECK-NEXT:  	vfmadd231ss -0x7fc(%r10,%r8,4),%xmm5,%xmm8
// CHECK-NEXT:  	vfmadd231ss -0x3fc(%r10,%r8,4),%xmm6,%xmm8
// CHECK-NEXT:  	vfmadd231ss 0x4(%r10,%r8,4),%xmm7,%xmm8
// CHECK-NEXT:  	vmovss %xmm8,0x4(%r13,%r8,4)
// CHECK-NEXT:  	inc    %r8
// CHECK-NEXT:  	cmp    $0x3f,%r8
// CHECK-NEXT:  	jb     <myfun+0x160>
// CHECK-NEXT:  	mov    $0xffffffffffffffff,%r8
// CHECK-NEXT:  	nopl   0x0(%rax,%rax,1)
// CHECK-NEXT:  	vmovss -0x1bfc(%r11,%r8,4),%xmm8
// CHECK-NEXT:  	vfmadd213ss 0x4(%rax,%r8,4),%xmm0,%xmm8
// CHECK-NEXT:  	vfmadd231ss -0x17fc(%r11,%r8,4),%xmm1,%xmm8
// CHECK-NEXT:  	vfmadd231ss -0x13fc(%r11,%r8,4),%xmm2,%xmm8
// CHECK-NEXT:  	vfmadd231ss -0xffc(%r11,%r8,4),%xmm3,%xmm8
// CHECK-NEXT:  	vfmadd231ss -0xbfc(%r11,%r8,4),%xmm4,%xmm8
// CHECK-NEXT:  	vfmadd231ss -0x7fc(%r11,%r8,4),%xmm5,%xmm8
// CHECK-NEXT:  	vfmadd231ss -0x3fc(%r11,%r8,4),%xmm6,%xmm8
// CHECK-NEXT:  	vfmadd231ss 0x4(%r11,%r8,4),%xmm7,%xmm8
// CHECK-NEXT:  	vmovss %xmm8,0x4(%rax,%r8,4)
// CHECK-NEXT:  	inc    %r8
// CHECK-NEXT:  	cmp    $0x3f,%r8
// CHECK-NEXT:  	jb     <myfun+0x1d0>
// CHECK-NEXT:  	mov    $0xffffffffffffffff,%r8
// CHECK-NEXT:  	nopl   0x0(%rax,%rax,1)
// CHECK-NEXT:  	vmovss -0x1bfc(%rbx,%r8,4),%xmm8
// CHECK-NEXT:  	vfmadd213ss 0x4(%r14,%r8,4),%xmm0,%xmm8
// CHECK-NEXT:  	vfmadd231ss -0x17fc(%rbx,%r8,4),%xmm1,%xmm8
// CHECK-NEXT:  	vfmadd231ss -0x13fc(%rbx,%r8,4),%xmm2,%xmm8
// CHECK-NEXT:  	vfmadd231ss -0xffc(%rbx,%r8,4),%xmm3,%xmm8
// CHECK-NEXT:  	vfmadd231ss -0xbfc(%rbx,%r8,4),%xmm4,%xmm8
// CHECK-NEXT:  	vfmadd231ss -0x7fc(%rbx,%r8,4),%xmm5,%xmm8
// CHECK-NEXT:  	vfmadd231ss -0x3fc(%rbx,%r8,4),%xmm6,%xmm8
// CHECK-NEXT:  	vfmadd231ss 0x4(%rbx,%r8,4),%xmm7,%xmm8
// CHECK-NEXT:  	vmovss %xmm8,0x4(%r14,%r8,4)
// CHECK-NEXT:  	inc    %r8
// CHECK-NEXT:  	cmp    $0x3f,%r8
// CHECK-NEXT:  	jb     <myfun+0x240>
// CHECK-NEXT:  	lea    0x8(%rdx),%r8
// CHECK-NEXT:  	add    $0x2000,%r9
// CHECK-NEXT:  	add    $0x2000,%r10
// CHECK-NEXT:  	add    $0x2000,%r11
// CHECK-NEXT:  	add    $0x2000,%rbx
// CHECK-NEXT:  	cmp    $0x1f8,%rdx
// CHECK-NEXT:  	mov    %r8,%rdx
// CHECK-NEXT:  	jb     <myfun+0xb0>
// CHECK-NEXT:  	mov    $0x1,%r14d
// CHECK-NEXT:  	test   $0x1,%bpl
// CHECK-NEXT:  	mov    $0x0,%ebp
// CHECK-NEXT:  	mov    %rcx,%rdx
// CHECK-NEXT:  	jne    <myfun+0x70>
// CHECK-NEXT:  	add    $0x800,%rdx
// CHECK-NEXT:  	add    $0x101c,%rdi
// CHECK-NEXT:  	mov    -0x38(%rsp),%rbx
// CHECK-NEXT:  	add    $0x1c00,%rbx
// CHECK-NEXT:  	xor    %eax,%eax
// CHECK-NEXT:  	nopl   0x0(%rax,%rax,1)
// CHECK-NEXT:  	mov    %rax,%rcx
// CHECK-NEXT:  	shl    $0xa,%rcx
// CHECK-NEXT:  	add    %rdx,%rcx
// CHECK-NEXT:  	mov    %rbx,%r8
// CHECK-NEXT:  	xor    %r9d,%r9d
// CHECK-NEXT:  	vmovups 0xe0(%rcx,%r9,4),%ymm0
// CHECK-NEXT:  	vmovups 0xc0(%rcx,%r9,4),%ymm1
// CHECK-NEXT:  	vmovups 0xa0(%rcx,%r9,4),%ymm2
// CHECK-NEXT:  	vmovups 0x80(%rcx,%r9,4),%ymm3
// CHECK-NEXT:  	vmovups (%rcx,%r9,4),%ymm4
// CHECK-NEXT:  	vmovups 0x20(%rcx,%r9,4),%ymm5
// CHECK-NEXT:  	vmovups 0x40(%rcx,%r9,4),%ymm6
// CHECK-NEXT:  	vmovups 0x60(%rcx,%r9,4),%ymm7
// CHECK-NEXT:  	mov    $0xfffffffffffffff8,%r10
// CHECK-NEXT:  	mov    %r8,%r11
// CHECK-NEXT:  	nopl   (%rax)
// CHECK-NEXT:  	vbroadcastss 0x4(%rdi,%r10,4),%ymm8
// CHECK-NEXT:  	vfmadd231ps -0x1b20(%r11),%ymm8,%ymm0
// CHECK-NEXT:  	vfmadd231ps -0x1b40(%r11),%ymm8,%ymm1
// CHECK-NEXT:  	vfmadd231ps -0x1b60(%r11),%ymm8,%ymm2
// CHECK-NEXT:  	vfmadd231ps -0x1b80(%r11),%ymm8,%ymm3
// CHECK-NEXT:  	vfmadd231ps -0x1ba0(%r11),%ymm8,%ymm7
// CHECK-NEXT:  	vfmadd231ps -0x1bc0(%r11),%ymm8,%ymm6
// CHECK-NEXT:  	vfmadd231ps -0x1c00(%r11),%ymm8,%ymm4
// CHECK-NEXT:  	vfmadd231ps -0x1be0(%r11),%ymm8,%ymm5
// CHECK-NEXT:  	vbroadcastss 0x8(%rdi,%r10,4),%ymm8
// CHECK-NEXT:  	vfmadd231ps -0x17e0(%r11),%ymm8,%ymm5
// CHECK-NEXT:  	vfmadd231ps -0x1800(%r11),%ymm8,%ymm4
// CHECK-NEXT:  	vfmadd231ps -0x17c0(%r11),%ymm8,%ymm6
// CHECK-NEXT:  	vfmadd231ps -0x17a0(%r11),%ymm8,%ymm7
// CHECK-NEXT:  	vfmadd231ps -0x1780(%r11),%ymm8,%ymm3
// CHECK-NEXT:  	vfmadd231ps -0x1760(%r11),%ymm8,%ymm2
// CHECK-NEXT:  	vfmadd231ps -0x1740(%r11),%ymm8,%ymm1
// CHECK-NEXT:  	vfmadd231ps -0x1720(%r11),%ymm8,%ymm0
// CHECK-NEXT:  	vbroadcastss 0xc(%rdi,%r10,4),%ymm8
// CHECK-NEXT:  	vfmadd231ps -0x1320(%r11),%ymm8,%ymm0
// CHECK-NEXT:  	vfmadd231ps -0x1340(%r11),%ymm8,%ymm1
// CHECK-NEXT:  	vfmadd231ps -0x1360(%r11),%ymm8,%ymm2
// CHECK-NEXT:  	vfmadd231ps -0x1380(%r11),%ymm8,%ymm3
// CHECK-NEXT:  	vfmadd231ps -0x13a0(%r11),%ymm8,%ymm7
// CHECK-NEXT:  	vfmadd231ps -0x13c0(%r11),%ymm8,%ymm6
// CHECK-NEXT:  	vfmadd231ps -0x1400(%r11),%ymm8,%ymm4
// CHECK-NEXT:  	vfmadd231ps -0x13e0(%r11),%ymm8,%ymm5
// CHECK-NEXT:  	vbroadcastss 0x10(%rdi,%r10,4),%ymm8
// CHECK-NEXT:  	vfmadd231ps -0xfe0(%r11),%ymm8,%ymm5
// CHECK-NEXT:  	vfmadd231ps -0x1000(%r11),%ymm8,%ymm4
// CHECK-NEXT:  	vfmadd231ps -0xfc0(%r11),%ymm8,%ymm6
// CHECK-NEXT:  	vfmadd231ps -0xfa0(%r11),%ymm8,%ymm7
// CHECK-NEXT:  	vfmadd231ps -0xf80(%r11),%ymm8,%ymm3
// CHECK-NEXT:  	vfmadd231ps -0xf60(%r11),%ymm8,%ymm2
// CHECK-NEXT:  	vfmadd231ps -0xf40(%r11),%ymm8,%ymm1
// CHECK-NEXT:  	vfmadd231ps -0xf20(%r11),%ymm8,%ymm0
// CHECK-NEXT:  	vbroadcastss 0x14(%rdi,%r10,4),%ymm8
// CHECK-NEXT:  	vfmadd231ps -0xb20(%r11),%ymm8,%ymm0
// CHECK-NEXT:  	vfmadd231ps -0xb40(%r11),%ymm8,%ymm1
// CHECK-NEXT:  	vfmadd231ps -0xb60(%r11),%ymm8,%ymm2
// CHECK-NEXT:  	vfmadd231ps -0xb80(%r11),%ymm8,%ymm3
// CHECK-NEXT:  	vfmadd231ps -0xba0(%r11),%ymm8,%ymm7
// CHECK-NEXT:  	vfmadd231ps -0xbc0(%r11),%ymm8,%ymm6
// CHECK-NEXT:  	vfmadd231ps -0xc00(%r11),%ymm8,%ymm4
// CHECK-NEXT:  	vfmadd231ps -0xbe0(%r11),%ymm8,%ymm5
// CHECK-NEXT:  	vbroadcastss 0x18(%rdi,%r10,4),%ymm8
// CHECK-NEXT:  	vfmadd231ps -0x7e0(%r11),%ymm8,%ymm5
// CHECK-NEXT:  	vfmadd231ps -0x800(%r11),%ymm8,%ymm4
// CHECK-NEXT:  	vfmadd231ps -0x7c0(%r11),%ymm8,%ymm6
// CHECK-NEXT:  	vfmadd231ps -0x7a0(%r11),%ymm8,%ymm7
// CHECK-NEXT:  	vfmadd231ps -0x780(%r11),%ymm8,%ymm3
// CHECK-NEXT:  	vfmadd231ps -0x760(%r11),%ymm8,%ymm2
// CHECK-NEXT:  	vfmadd231ps -0x740(%r11),%ymm8,%ymm1
// CHECK-NEXT:  	vfmadd231ps -0x720(%r11),%ymm8,%ymm0
// CHECK-NEXT:  	vbroadcastss 0x1c(%rdi,%r10,4),%ymm8
// CHECK-NEXT:  	vfmadd231ps -0x320(%r11),%ymm8,%ymm0
// CHECK-NEXT:  	vfmadd231ps -0x340(%r11),%ymm8,%ymm1
// CHECK-NEXT:  	vfmadd231ps -0x360(%r11),%ymm8,%ymm2
// CHECK-NEXT:  	vfmadd231ps -0x380(%r11),%ymm8,%ymm3
// CHECK-NEXT:  	vfmadd231ps -0x3a0(%r11),%ymm8,%ymm7
// CHECK-NEXT:  	vfmadd231ps -0x3c0(%r11),%ymm8,%ymm6
// CHECK-NEXT:  	vfmadd231ps -0x400(%r11),%ymm8,%ymm4
// CHECK-NEXT:  	vfmadd231ps -0x3e0(%r11),%ymm8,%ymm5
// CHECK-NEXT:  	vbroadcastss 0x20(%rdi,%r10,4),%ymm8
// CHECK-NEXT:  	vfmadd231ps 0x20(%r11),%ymm8,%ymm5
// CHECK-NEXT:  	vfmadd231ps (%r11),%ymm8,%ymm4
// CHECK-NEXT:  	vfmadd231ps 0x40(%r11),%ymm8,%ymm6
// CHECK-NEXT:  	vfmadd231ps 0x60(%r11),%ymm8,%ymm7
// CHECK-NEXT:  	vfmadd231ps 0x80(%r11),%ymm8,%ymm3
// CHECK-NEXT:  	vfmadd231ps 0xa0(%r11),%ymm8,%ymm2
// CHECK-NEXT:  	vfmadd231ps 0xc0(%r11),%ymm8,%ymm1
// CHECK-NEXT:  	vfmadd231ps 0xe0(%r11),%ymm8,%ymm0
// CHECK-NEXT:  	add    $0x8,%r10
// CHECK-NEXT:  	add    $0x2000,%r11
// CHECK-NEXT:  	cmp    $0x1f8,%r10
// CHECK-NEXT:  	jb     <myfun+0x370>
// CHECK-NEXT:  	vmovups %ymm4,(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %ymm5,0x20(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %ymm6,0x40(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %ymm7,0x60(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %ymm3,0x80(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %ymm2,0xa0(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %ymm1,0xc0(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %ymm0,0xe0(%rcx,%r9,4)
// CHECK-NEXT:  	lea    0x40(%r9),%rsi
// CHECK-NEXT:  	add    $0x100,%r8
// CHECK-NEXT:  	cmp    $0xc0,%r9
// CHECK-NEXT:  	mov    %rsi,%r9
// CHECK-NEXT:  	jb     <myfun+0x320>
// CHECK-NEXT:  	lea    0x1(%rax),%rcx
// CHECK-NEXT:  	add    $0x800,%rdi
// CHECK-NEXT:  	cmp    $0xff,%rax
// CHECK-NEXT:  	mov    %rcx,%rax
// CHECK-NEXT:  	jb     <myfun+0x310>
// CHECK-NEXT:  	pop    %rbx
// CHECK-NEXT:  	pop    %r12
// CHECK-NEXT:  	pop    %r13
// CHECK-NEXT:  	pop    %r14
// CHECK-NEXT:  	pop    %r15
// CHECK-NEXT:  	pop    %rbp
// CHECK-NEXT:  	vzeroupper
// CHECK-NEXT:  	ret
