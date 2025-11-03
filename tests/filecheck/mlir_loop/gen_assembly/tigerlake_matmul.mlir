// RUN: mlir-loop --no-alias --arch x86-64 --cpu tigerlake --print-assembly --hide-jumps %s 2>&1 | grep -v '\(nop\|ret\)' | filecheck %s
// UNSUPPORTED: mlir-target=c
// Assembly output will differ a bit when using C.

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
                "K#8"= {"unroll"},
                  "J#64" = {"vectorize"}
      }
    }
    ins(%A, %B : memref<256x512xf32>, memref<512x256xf32>)
    outs(%C : memref<256x256xf32>)
  return
}
// CHECK:       Disassembly of section .text:
// CHECK-NEXT:  
// CHECK-NEXT:  <myfun>:
// CHECK-NEXT:  	push   %rbx
// CHECK-NEXT:  	lea    0x1c00(%rsi),%rax
// CHECK-NEXT:  	add    $0x1c,%rdi
// CHECK-NEXT:  	lea    0x1d00(%rsi),%rcx
// CHECK-NEXT:  	lea    0x1e00(%rsi),%r8
// CHECK-NEXT:  	add    $0x1f00,%rsi
// CHECK-NEXT:  	xor    %r9d,%r9d
// CHECK-NEXT:  	mov    %r9,%r10
// CHECK-NEXT:  	shl    $0xa,%r10
// CHECK-NEXT:  	vmovups (%rdx,%r10,1),%zmm0
// CHECK-NEXT:  	vmovups 0x40(%rdx,%r10,1),%zmm1
// CHECK-NEXT:  	vmovups 0x80(%rdx,%r10,1),%zmm2
// CHECK-NEXT:  	vmovups 0xc0(%rdx,%r10,1),%zmm3
// CHECK-NEXT:  	mov    $0xfffffffffffffff8,%r11
// CHECK-NEXT:  	mov    %rax,%rbx
// CHECK-NEXT:  	vbroadcastss 0x4(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0x1b40(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	vfmadd231ps -0x1b80(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0x1c00(%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0x1bc0(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vbroadcastss 0x8(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0x17c0(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vfmadd231ps -0x1800(%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0x1780(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0x1740(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	vbroadcastss 0xc(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0x1340(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	vfmadd231ps -0x1380(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0x1400(%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0x13c0(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vbroadcastss 0x10(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0xfc0(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vfmadd231ps -0x1000(%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0xf80(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0xf40(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	vbroadcastss 0x14(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0xb40(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	vfmadd231ps -0xb80(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0xc00(%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0xbc0(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vbroadcastss 0x18(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0x7c0(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vfmadd231ps -0x800(%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0x780(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0x740(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	vbroadcastss 0x1c(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0x340(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	vfmadd231ps -0x380(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0x400(%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0x3c0(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vbroadcastss 0x20(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps 0x40(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vfmadd231ps (%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps 0x80(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps 0xc0(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	add    $0x8,%r11
// CHECK-NEXT:  	add    $0x2000,%rbx
// CHECK-NEXT:  	cmp    $0x1f8,%r11
// CHECK-NEXT:  	jb     <myfun+0x60>
// CHECK-NEXT:  	vmovups %zmm0,(%rdx,%r10,1)
// CHECK-NEXT:  	vmovups %zmm1,0x40(%rdx,%r10,1)
// CHECK-NEXT:  	vmovups %zmm2,0x80(%rdx,%r10,1)
// CHECK-NEXT:  	vmovups %zmm3,0xc0(%rdx,%r10,1)
// CHECK-NEXT:  	vmovups 0x100(%rdx,%r10,1),%zmm0
// CHECK-NEXT:  	vmovups 0x140(%rdx,%r10,1),%zmm1
// CHECK-NEXT:  	vmovups 0x180(%rdx,%r10,1),%zmm2
// CHECK-NEXT:  	vmovups 0x1c0(%rdx,%r10,1),%zmm3
// CHECK-NEXT:  	mov    $0xfffffffffffffff8,%r11
// CHECK-NEXT:  	mov    %rcx,%rbx
// CHECK-NEXT:  	vbroadcastss 0x4(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0x1b40(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	vfmadd231ps -0x1b80(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0x1c00(%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0x1bc0(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vbroadcastss 0x8(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0x17c0(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vfmadd231ps -0x1800(%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0x1780(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0x1740(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	vbroadcastss 0xc(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0x1340(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	vfmadd231ps -0x1380(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0x1400(%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0x13c0(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vbroadcastss 0x10(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0xfc0(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vfmadd231ps -0x1000(%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0xf80(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0xf40(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	vbroadcastss 0x14(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0xb40(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	vfmadd231ps -0xb80(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0xc00(%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0xbc0(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vbroadcastss 0x18(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0x7c0(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vfmadd231ps -0x800(%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0x780(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0x740(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	vbroadcastss 0x1c(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0x340(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	vfmadd231ps -0x380(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0x400(%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0x3c0(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vbroadcastss 0x20(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps 0x40(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vfmadd231ps (%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps 0x80(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps 0xc0(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	add    $0x8,%r11
// CHECK-NEXT:  	add    $0x2000,%rbx
// CHECK-NEXT:  	cmp    $0x1f8,%r11
// CHECK-NEXT:  	jb     <myfun+0x1e0>
// CHECK-NEXT:  	vmovups %zmm0,0x100(%rdx,%r10,1)
// CHECK-NEXT:  	vmovups %zmm1,0x140(%rdx,%r10,1)
// CHECK-NEXT:  	vmovups %zmm2,0x180(%rdx,%r10,1)
// CHECK-NEXT:  	vmovups %zmm3,0x1c0(%rdx,%r10,1)
// CHECK-NEXT:  	vmovups 0x200(%rdx,%r10,1),%zmm0
// CHECK-NEXT:  	vmovups 0x240(%rdx,%r10,1),%zmm1
// CHECK-NEXT:  	vmovups 0x280(%rdx,%r10,1),%zmm2
// CHECK-NEXT:  	vmovups 0x2c0(%rdx,%r10,1),%zmm3
// CHECK-NEXT:  	mov    $0xfffffffffffffff8,%r11
// CHECK-NEXT:  	mov    %r8,%rbx
// CHECK-NEXT:  	vbroadcastss 0x4(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0x1b40(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	vfmadd231ps -0x1b80(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0x1c00(%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0x1bc0(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vbroadcastss 0x8(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0x17c0(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vfmadd231ps -0x1800(%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0x1780(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0x1740(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	vbroadcastss 0xc(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0x1340(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	vfmadd231ps -0x1380(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0x1400(%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0x13c0(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vbroadcastss 0x10(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0xfc0(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vfmadd231ps -0x1000(%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0xf80(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0xf40(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	vbroadcastss 0x14(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0xb40(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	vfmadd231ps -0xb80(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0xc00(%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0xbc0(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vbroadcastss 0x18(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0x7c0(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vfmadd231ps -0x800(%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0x780(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0x740(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	vbroadcastss 0x1c(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0x340(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	vfmadd231ps -0x380(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0x400(%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0x3c0(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vbroadcastss 0x20(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps 0x40(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vfmadd231ps (%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps 0x80(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps 0xc0(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	add    $0x8,%r11
// CHECK-NEXT:  	add    $0x2000,%rbx
// CHECK-NEXT:  	cmp    $0x1f8,%r11
// CHECK-NEXT:  	jb     <myfun+0x370>
// CHECK-NEXT:  	vmovups %zmm0,0x200(%rdx,%r10,1)
// CHECK-NEXT:  	vmovups %zmm1,0x240(%rdx,%r10,1)
// CHECK-NEXT:  	vmovups %zmm2,0x280(%rdx,%r10,1)
// CHECK-NEXT:  	vmovups %zmm3,0x2c0(%rdx,%r10,1)
// CHECK-NEXT:  	vmovups 0x300(%rdx,%r10,1),%zmm0
// CHECK-NEXT:  	vmovups 0x340(%rdx,%r10,1),%zmm1
// CHECK-NEXT:  	vmovups 0x380(%rdx,%r10,1),%zmm2
// CHECK-NEXT:  	vmovups 0x3c0(%rdx,%r10,1),%zmm3
// CHECK-NEXT:  	mov    $0xfffffffffffffff8,%r11
// CHECK-NEXT:  	mov    %rsi,%rbx
// CHECK-NEXT:  	vbroadcastss 0x4(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0x1b40(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	vfmadd231ps -0x1b80(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0x1c00(%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0x1bc0(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vbroadcastss 0x8(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0x17c0(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vfmadd231ps -0x1800(%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0x1780(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0x1740(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	vbroadcastss 0xc(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0x1340(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	vfmadd231ps -0x1380(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0x1400(%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0x13c0(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vbroadcastss 0x10(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0xfc0(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vfmadd231ps -0x1000(%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0xf80(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0xf40(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	vbroadcastss 0x14(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0xb40(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	vfmadd231ps -0xb80(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0xc00(%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0xbc0(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vbroadcastss 0x18(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0x7c0(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vfmadd231ps -0x800(%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0x780(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0x740(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	vbroadcastss 0x1c(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0x340(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	vfmadd231ps -0x380(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0x400(%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0x3c0(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vbroadcastss 0x20(%rdi,%r11,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps 0x40(%rbx),%zmm4,%zmm1
// CHECK-NEXT:  	vfmadd231ps (%rbx),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps 0x80(%rbx),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps 0xc0(%rbx),%zmm4,%zmm3
// CHECK-NEXT:  	add    $0x8,%r11
// CHECK-NEXT:  	add    $0x2000,%rbx
// CHECK-NEXT:  	cmp    $0x1f8,%r11
// CHECK-NEXT:  	jb     <myfun+0x500>
// CHECK-NEXT:  	vmovups %zmm0,0x300(%rdx,%r10,1)
// CHECK-NEXT:  	vmovups %zmm1,0x340(%rdx,%r10,1)
// CHECK-NEXT:  	vmovups %zmm2,0x380(%rdx,%r10,1)
// CHECK-NEXT:  	vmovups %zmm3,0x3c0(%rdx,%r10,1)
// CHECK-NEXT:  	add    $0x800,%rdi
// CHECK-NEXT:  	cmp    $0xff,%r9
// CHECK-NEXT:  	lea    0x1(%r9),%r9
// CHECK-NEXT:  	jb     <myfun+0x30>
// CHECK-NEXT:  	pop    %rbx
// CHECK-NEXT:  	vzeroupper 
