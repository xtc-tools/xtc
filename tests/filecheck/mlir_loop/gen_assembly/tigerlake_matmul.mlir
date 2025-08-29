// RUN: mlir-loop --no-alias --arch x86-64 --cpu tigerlake --print-assembly --hide-jumps %s 2>&1 | filecheck %s
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
// CHECK:       <myfun>:
// CHECK-NEXT:  	add    $0x1c00,%rsi
// CHECK-NEXT:  	add    $0x1c,%rdi
// CHECK-NEXT:  	xor    %eax,%eax
// CHECK-NEXT:  	nopl   (%rax)
// CHECK-NEXT:  	mov    %rax,%rcx
// CHECK-NEXT:  	shl    $0xa,%rcx
// CHECK-NEXT:  	add    %rdx,%rcx
// CHECK-NEXT:  	mov    %rsi,%r8
// CHECK-NEXT:  	xor    %r9d,%r9d
// CHECK-NEXT:  	vmovups (%rcx,%r9,4),%zmm0
// CHECK-NEXT:  	vmovups 0x40(%rcx,%r9,4),%zmm1
// CHECK-NEXT:  	vmovups 0x80(%rcx,%r9,4),%zmm2
// CHECK-NEXT:  	vmovups 0xc0(%rcx,%r9,4),%zmm3
// CHECK-NEXT:  	mov    $0xfffffffffffffff8,%r10
// CHECK-NEXT:  	mov    %r8,%r11
// CHECK-NEXT:  	nopl   0x0(%rax)
// CHECK-NEXT:  	vbroadcastss 0x4(%rdi,%r10,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0x1b40(%r11),%zmm4,%zmm3
// CHECK-NEXT:  	vfmadd231ps -0x1b80(%r11),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0x1c00(%r11),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0x1bc0(%r11),%zmm4,%zmm1
// CHECK-NEXT:  	vbroadcastss 0x8(%rdi,%r10,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0x17c0(%r11),%zmm4,%zmm1
// CHECK-NEXT:  	vfmadd231ps -0x1800(%r11),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0x1780(%r11),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0x1740(%r11),%zmm4,%zmm3
// CHECK-NEXT:  	vbroadcastss 0xc(%rdi,%r10,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0x1340(%r11),%zmm4,%zmm3
// CHECK-NEXT:  	vfmadd231ps -0x1380(%r11),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0x1400(%r11),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0x13c0(%r11),%zmm4,%zmm1
// CHECK-NEXT:  	vbroadcastss 0x10(%rdi,%r10,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0xfc0(%r11),%zmm4,%zmm1
// CHECK-NEXT:  	vfmadd231ps -0x1000(%r11),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0xf80(%r11),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0xf40(%r11),%zmm4,%zmm3
// CHECK-NEXT:  	vbroadcastss 0x14(%rdi,%r10,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0xb40(%r11),%zmm4,%zmm3
// CHECK-NEXT:  	vfmadd231ps -0xb80(%r11),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0xc00(%r11),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0xbc0(%r11),%zmm4,%zmm1
// CHECK-NEXT:  	vbroadcastss 0x18(%rdi,%r10,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0x7c0(%r11),%zmm4,%zmm1
// CHECK-NEXT:  	vfmadd231ps -0x800(%r11),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0x780(%r11),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0x740(%r11),%zmm4,%zmm3
// CHECK-NEXT:  	vbroadcastss 0x1c(%rdi,%r10,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps -0x340(%r11),%zmm4,%zmm3
// CHECK-NEXT:  	vfmadd231ps -0x380(%r11),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps -0x400(%r11),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps -0x3c0(%r11),%zmm4,%zmm1
// CHECK-NEXT:  	vbroadcastss 0x20(%rdi,%r10,4),%zmm4
// CHECK-NEXT:  	vfmadd231ps 0x40(%r11),%zmm4,%zmm1
// CHECK-NEXT:  	vfmadd231ps (%r11),%zmm4,%zmm0
// CHECK-NEXT:  	vfmadd231ps 0x80(%r11),%zmm4,%zmm2
// CHECK-NEXT:  	vfmadd231ps 0xc0(%r11),%zmm4,%zmm3
// CHECK-NEXT:  	add    $0x8,%r10
// CHECK-NEXT:  	add    $0x2000,%r11
// CHECK-NEXT:  	cmp    $0x1f8,%r10
// CHECK-NEXT:  	jb     <myfun+0x50>
// CHECK-NEXT:  	vmovups %zmm0,(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %zmm1,0x40(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %zmm2,0x80(%rcx,%r9,4)
// CHECK-NEXT:  	vmovups %zmm3,0xc0(%rcx,%r9,4)
// CHECK-NEXT:  	lea    0x40(%r9),%r10
// CHECK-NEXT:  	add    $0x100,%r8
// CHECK-NEXT:  	cmp    $0xc0,%r9
// CHECK-NEXT:  	mov    %r10,%r9
// CHECK-NEXT:  	jb     <myfun+0x20>
// CHECK-NEXT:  	lea    0x1(%rax),%rcx
// CHECK-NEXT:  	add    $0x800,%rdi
// CHECK-NEXT:  	cmp    $0xff,%rax
// CHECK-NEXT:  	mov    %rcx,%rax
// CHECK-NEXT:  	jb     <myfun+0x10>
// CHECK-NEXT:  	vzeroupper
// CHECK-NEXT:  	ret
