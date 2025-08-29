// RUN: mlir-loop --no-alias --arch x86-64 --cpu skylake --print-assembly --hide-jumps %s 2>&1 | filecheck %s
// UNSUPPORTED: mlir-target=c
// Assembly output will differ a bit when using C.

func.func @myfun(
  %A: memref<256x512xf32>,
  %B: memref<512x256xf32>,
  %C: memref<256x256xf32>
) {
  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d2, d1)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  }
    ins(%A, %B : memref<256x512xf32>, memref<512x256xf32>)
    outs(%C : memref<256x256xf32>)
      attrs = {
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
    {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    }
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
// CHECK-NEXT:  	jb     <myfun+0x70>
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
// CHECK-NEXT:  	jb     <myfun+0x20>
// CHECK-NEXT:  	lea    0x1(%rax),%rcx
// CHECK-NEXT:  	add    $0x800,%rdi
// CHECK-NEXT:  	cmp    $0xff,%rax
// CHECK-NEXT:  	mov    %rcx,%rax
// CHECK-NEXT:  	jb     <myfun+0x10>
// CHECK-NEXT:  	vzeroupper
// CHECK-NEXT:  	ret
