// RUN: mlir-loop --no-alias --arch x86-64 --cpu skylake --print-assembly --hide-jumps %s 2>&1 | filecheck %s

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
// CHECK-NEXT:  	xor    %ecx,%ecx
// CHECK-NEXT:  	mov    %rsi,%rax
// CHECK-NEXT:  	data16 cs nopw 0x0(%rax,%rax,1)
// CHECK-NEXT:  	vmovss (%rdi,%rcx,4),%xmm0
// CHECK-NEXT:  	mov    $0xffffffffffffffff,%r8
// CHECK-NEXT:  	nopl   0x0(%rax)
// CHECK-NEXT:  	vmovss 0x4(%rax,%r8,4),%xmm1
// CHECK-NEXT:  	vfmadd213ss 0x4(%rdx,%r8,4),%xmm0,%xmm1
// CHECK-NEXT:  	vmovss %xmm1,0x4(%rdx,%r8,4)
// CHECK-NEXT:  	inc    %r8
// CHECK-NEXT:  	cmp    $0xff,%r8
// CHECK-NEXT:  	jb     <myfun+0x20>
// CHECK-NEXT:  	lea    0x1(%rcx),%r8
// CHECK-NEXT:  	add    $0x400,%rax
// CHECK-NEXT:  	cmp    $0x1ff,%rcx
// CHECK-NEXT:  	mov    %r8,%rcx
// CHECK-NEXT:  	jb     <myfun+0x10>
// CHECK-NEXT:  	xor    %ecx,%ecx
// CHECK-NEXT:  	mov    %rsi,%rax
// CHECK-NEXT:  	nopl   0x0(%rax)
// CHECK-NEXT:  	vmovss 0x800(%rdi,%rcx,4),%xmm0
// CHECK-NEXT:  	mov    $0xffffffffffffffff,%r8
// CHECK-NEXT:  	vmovss 0x4(%rax,%r8,4),%xmm1
// CHECK-NEXT:  	vfmadd213ss 0x404(%rdx,%r8,4),%xmm0,%xmm1
// CHECK-NEXT:  	vmovss %xmm1,0x404(%rdx,%r8,4)
// CHECK-NEXT:  	inc    %r8
// CHECK-NEXT:  	cmp    $0xff,%r8
// CHECK-NEXT:  	jb     <myfun+0x70>
// CHECK-NEXT:  	lea    0x1(%rcx),%r8
// CHECK-NEXT:  	add    $0x400,%rax
// CHECK-NEXT:  	cmp    $0x1ff,%rcx
// CHECK-NEXT:  	mov    %r8,%rcx
// CHECK-NEXT:  	jb     <myfun+0x60>
// CHECK-NEXT:  	add    $0x800,%rdx
// CHECK-NEXT:  	add    $0x101c,%rdi
// CHECK-NEXT:  	add    $0x1c00,%rsi
// CHECK-NEXT:  	xor    %eax,%eax
// CHECK-NEXT:  	data16 data16 cs nopw 0x0(%rax,%rax,1)
// CHECK-NEXT:  	mov    %rax,%rcx
// CHECK-NEXT:  	shl    $0xa,%rcx
// CHECK-NEXT:  	add    %rdx,%rcx
// CHECK-NEXT:  	mov    %rsi,%r8
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
// CHECK-NEXT:  	jb     <myfun+0x130>
// CHECK-NEXT:  	vmovups %ymm4,(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %ymm5,0x20(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %ymm6,0x40(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %ymm7,0x60(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %ymm3,0x80(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %ymm2,0xa0(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %ymm1,0xc0(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %ymm0,0xe0(%rcx,%r9,4)
// CHECK-NEXT:  	lea    0x40(%r9),%r10
// CHECK-NEXT:  	add    $0x100,%r8
// CHECK-NEXT:  	cmp    $0xc0,%r9
// CHECK-NEXT:  	mov    %r10,%r9
// CHECK-NEXT:  	jb     <myfun+0xe0>
// CHECK-NEXT:  	lea    0x1(%rax),%rcx
// CHECK-NEXT:  	add    $0x800,%rdi
// CHECK-NEXT:  	cmp    $0xff,%rax
// CHECK-NEXT:  	mov    %rcx,%rax
// CHECK-NEXT:  	jb     <myfun+0xd0>
// CHECK-NEXT:  	vzeroupper
// CHECK-NEXT:  	ret
