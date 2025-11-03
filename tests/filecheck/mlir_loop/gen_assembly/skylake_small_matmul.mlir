// RUN: mlir-loop --no-alias --arch x86-64 --cpu skylake --print-assembly --hide-jumps %s 2>&1 | grep -v '\(nop\|ret\)' | filecheck %s
// UNSUPPORTED: mlir-target=c
// Assembly output will differ a bit when using C.

func.func @myfun(
  %A: memref<8x8xf32>,
  %B: memref<8x8xf32>,
  %C: memref<8x8xf32>
) {
  linalg.matmul
    {
      loop.dims = ["I","J","K"],
      loop.schedule = {
        "I",
          "J",
            "K",
              "I#1" = {"unroll"},
                "K#8" = {"unroll"},
                  "J#8" = {"vectorize"}
      }
    }
    ins(%A, %B : memref<8x8xf32>, memref<8x8xf32>)
    outs(%C : memref<8x8xf32>)
  return
}
// CHECK:       Disassembly of section .text:
// CHECK-NEXT:  
// CHECK-NEXT:  <myfun>:
// CHECK-NEXT:  	vmovups 0xe0(%rsi),%ymm0
// CHECK-NEXT:  	vmovups 0xc0(%rsi),%ymm1
// CHECK-NEXT:  	vmovups 0xa0(%rsi),%ymm2
// CHECK-NEXT:  	vmovups 0x80(%rsi),%ymm3
// CHECK-NEXT:  	vmovups (%rsi),%ymm7
// CHECK-NEXT:  	vmovups 0x20(%rsi),%ymm6
// CHECK-NEXT:  	vmovups 0x40(%rsi),%ymm5
// CHECK-NEXT:  	vmovups 0x60(%rsi),%ymm4
// CHECK-NEXT:  	vbroadcastss (%rdi),%ymm8
// CHECK-NEXT:  	vfmadd213ps (%rdx),%ymm7,%ymm8
// CHECK-NEXT:  	vbroadcastss 0x4(%rdi),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm6,%ymm9
// CHECK-NEXT:  	vbroadcastss 0x8(%rdi),%ymm8
// CHECK-NEXT:  	vfmadd213ps %ymm9,%ymm5,%ymm8
// CHECK-NEXT:  	vbroadcastss 0xc(%rdi),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm4,%ymm9
// CHECK-NEXT:  	vbroadcastss 0x10(%rdi),%ymm8
// CHECK-NEXT:  	vfmadd213ps %ymm9,%ymm3,%ymm8
// CHECK-NEXT:  	vbroadcastss 0x14(%rdi),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm2,%ymm9
// CHECK-NEXT:  	vbroadcastss 0x18(%rdi),%ymm8
// CHECK-NEXT:  	vfmadd213ps %ymm9,%ymm1,%ymm8
// CHECK-NEXT:  	vbroadcastss 0x1c(%rdi),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm0,%ymm9
// CHECK-NEXT:  	vmovups %ymm9,(%rdx)
// CHECK-NEXT:  	vbroadcastss 0x20(%rdi),%ymm8
// CHECK-NEXT:  	vfmadd213ps 0x20(%rdx),%ymm7,%ymm8
// CHECK-NEXT:  	vbroadcastss 0x24(%rdi),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm6,%ymm9
// CHECK-NEXT:  	vbroadcastss 0x28(%rdi),%ymm8
// CHECK-NEXT:  	vfmadd213ps %ymm9,%ymm5,%ymm8
// CHECK-NEXT:  	vbroadcastss 0x2c(%rdi),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm4,%ymm9
// CHECK-NEXT:  	vbroadcastss 0x30(%rdi),%ymm8
// CHECK-NEXT:  	vfmadd213ps %ymm9,%ymm3,%ymm8
// CHECK-NEXT:  	vbroadcastss 0x34(%rdi),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm2,%ymm9
// CHECK-NEXT:  	vbroadcastss 0x38(%rdi),%ymm8
// CHECK-NEXT:  	vfmadd213ps %ymm9,%ymm1,%ymm8
// CHECK-NEXT:  	vbroadcastss 0x3c(%rdi),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm0,%ymm9
// CHECK-NEXT:  	vmovups %ymm9,0x20(%rdx)
// CHECK-NEXT:  	vbroadcastss 0x40(%rdi),%ymm8
// CHECK-NEXT:  	vfmadd213ps 0x40(%rdx),%ymm7,%ymm8
// CHECK-NEXT:  	vbroadcastss 0x44(%rdi),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm6,%ymm9
// CHECK-NEXT:  	vbroadcastss 0x48(%rdi),%ymm8
// CHECK-NEXT:  	vfmadd213ps %ymm9,%ymm5,%ymm8
// CHECK-NEXT:  	vbroadcastss 0x4c(%rdi),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm4,%ymm9
// CHECK-NEXT:  	vbroadcastss 0x50(%rdi),%ymm8
// CHECK-NEXT:  	vfmadd213ps %ymm9,%ymm3,%ymm8
// CHECK-NEXT:  	vbroadcastss 0x54(%rdi),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm2,%ymm9
// CHECK-NEXT:  	vbroadcastss 0x58(%rdi),%ymm8
// CHECK-NEXT:  	vfmadd213ps %ymm9,%ymm1,%ymm8
// CHECK-NEXT:  	vbroadcastss 0x5c(%rdi),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm0,%ymm9
// CHECK-NEXT:  	vmovups %ymm9,0x40(%rdx)
// CHECK-NEXT:  	vbroadcastss 0x60(%rdi),%ymm8
// CHECK-NEXT:  	vfmadd213ps 0x60(%rdx),%ymm7,%ymm8
// CHECK-NEXT:  	vbroadcastss 0x64(%rdi),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm6,%ymm9
// CHECK-NEXT:  	vbroadcastss 0x68(%rdi),%ymm8
// CHECK-NEXT:  	vfmadd213ps %ymm9,%ymm5,%ymm8
// CHECK-NEXT:  	vbroadcastss 0x6c(%rdi),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm4,%ymm9
// CHECK-NEXT:  	vbroadcastss 0x70(%rdi),%ymm8
// CHECK-NEXT:  	vfmadd213ps %ymm9,%ymm3,%ymm8
// CHECK-NEXT:  	vbroadcastss 0x74(%rdi),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm2,%ymm9
// CHECK-NEXT:  	vbroadcastss 0x78(%rdi),%ymm8
// CHECK-NEXT:  	vfmadd213ps %ymm9,%ymm1,%ymm8
// CHECK-NEXT:  	vbroadcastss 0x7c(%rdi),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm0,%ymm9
// CHECK-NEXT:  	vmovups %ymm9,0x60(%rdx)
// CHECK-NEXT:  	vbroadcastss 0x80(%rdi),%ymm8
// CHECK-NEXT:  	vfmadd213ps 0x80(%rdx),%ymm7,%ymm8
// CHECK-NEXT:  	vbroadcastss 0x84(%rdi),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm6,%ymm9
// CHECK-NEXT:  	vbroadcastss 0x88(%rdi),%ymm8
// CHECK-NEXT:  	vfmadd213ps %ymm9,%ymm5,%ymm8
// CHECK-NEXT:  	vbroadcastss 0x8c(%rdi),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm4,%ymm9
// CHECK-NEXT:  	vbroadcastss 0x90(%rdi),%ymm8
// CHECK-NEXT:  	vfmadd213ps %ymm9,%ymm3,%ymm8
// CHECK-NEXT:  	vbroadcastss 0x94(%rdi),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm2,%ymm9
// CHECK-NEXT:  	vbroadcastss 0x98(%rdi),%ymm8
// CHECK-NEXT:  	vfmadd213ps %ymm9,%ymm1,%ymm8
// CHECK-NEXT:  	vbroadcastss 0x9c(%rdi),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm0,%ymm9
// CHECK-NEXT:  	vmovups %ymm9,0x80(%rdx)
// CHECK-NEXT:  	vbroadcastss 0xa0(%rdi),%ymm8
// CHECK-NEXT:  	vfmadd213ps 0xa0(%rdx),%ymm7,%ymm8
// CHECK-NEXT:  	vbroadcastss 0xa4(%rdi),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm6,%ymm9
// CHECK-NEXT:  	vbroadcastss 0xa8(%rdi),%ymm8
// CHECK-NEXT:  	vfmadd213ps %ymm9,%ymm5,%ymm8
// CHECK-NEXT:  	vbroadcastss 0xac(%rdi),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm4,%ymm9
// CHECK-NEXT:  	vbroadcastss 0xb0(%rdi),%ymm8
// CHECK-NEXT:  	vfmadd213ps %ymm9,%ymm3,%ymm8
// CHECK-NEXT:  	vbroadcastss 0xb4(%rdi),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm2,%ymm9
// CHECK-NEXT:  	vbroadcastss 0xb8(%rdi),%ymm8
// CHECK-NEXT:  	vfmadd213ps %ymm9,%ymm1,%ymm8
// CHECK-NEXT:  	vbroadcastss 0xbc(%rdi),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm0,%ymm9
// CHECK-NEXT:  	vmovups %ymm9,0xa0(%rdx)
// CHECK-NEXT:  	vbroadcastss 0xc0(%rdi),%ymm8
// CHECK-NEXT:  	vfmadd213ps 0xc0(%rdx),%ymm7,%ymm8
// CHECK-NEXT:  	vbroadcastss 0xc4(%rdi),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm6,%ymm9
// CHECK-NEXT:  	vbroadcastss 0xc8(%rdi),%ymm8
// CHECK-NEXT:  	vfmadd213ps %ymm9,%ymm5,%ymm8
// CHECK-NEXT:  	vbroadcastss 0xcc(%rdi),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm4,%ymm9
// CHECK-NEXT:  	vbroadcastss 0xd0(%rdi),%ymm8
// CHECK-NEXT:  	vfmadd213ps %ymm9,%ymm3,%ymm8
// CHECK-NEXT:  	vbroadcastss 0xd4(%rdi),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm2,%ymm9
// CHECK-NEXT:  	vbroadcastss 0xd8(%rdi),%ymm8
// CHECK-NEXT:  	vfmadd213ps %ymm9,%ymm1,%ymm8
// CHECK-NEXT:  	vbroadcastss 0xdc(%rdi),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm0,%ymm9
// CHECK-NEXT:  	vmovups %ymm9,0xc0(%rdx)
// CHECK-NEXT:  	vbroadcastss 0xe0(%rdi),%ymm8
// CHECK-NEXT:  	vfmadd213ps 0xe0(%rdx),%ymm7,%ymm8
// CHECK-NEXT:  	vbroadcastss 0xe4(%rdi),%ymm7
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm6,%ymm7
// CHECK-NEXT:  	vbroadcastss 0xe8(%rdi),%ymm6
// CHECK-NEXT:  	vfmadd213ps %ymm7,%ymm5,%ymm6
// CHECK-NEXT:  	vbroadcastss 0xec(%rdi),%ymm5
// CHECK-NEXT:  	vfmadd213ps %ymm6,%ymm4,%ymm5
// CHECK-NEXT:  	vbroadcastss 0xf0(%rdi),%ymm4
// CHECK-NEXT:  	vfmadd213ps %ymm5,%ymm3,%ymm4
// CHECK-NEXT:  	vbroadcastss 0xf4(%rdi),%ymm3
// CHECK-NEXT:  	vfmadd213ps %ymm4,%ymm2,%ymm3
// CHECK-NEXT:  	vbroadcastss 0xf8(%rdi),%ymm2
// CHECK-NEXT:  	vfmadd213ps %ymm3,%ymm1,%ymm2
// CHECK-NEXT:  	vbroadcastss 0xfc(%rdi),%ymm1
// CHECK-NEXT:  	vfmadd213ps %ymm2,%ymm0,%ymm1
// CHECK-NEXT:  	vmovups %ymm1,0xe0(%rdx)
// CHECK-NEXT:  	vzeroupper 
