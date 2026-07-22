// RUN: mlir-loop --no-alias --arch x86-64 --cpu skylake --print-assembly --hide-jumps %s 2>&1 | grep -v '\(nop\|ret\)' | filecheck %s
// REQUIRES: mlir-target=llvmir
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
// CHECK-NEXT:  	vmovups (%rsi),%ymm0
// CHECK-NEXT:  	vmovups 0x20(%rsi),%ymm1
// CHECK-NEXT:  	vmovups 0x40(%rsi),%ymm2
// CHECK-NEXT:  	vmovups 0x60(%rsi),%ymm3
// CHECK-NEXT:  	vmovups 0x80(%rsi),%ymm4
// CHECK-NEXT:  	vmovups 0xa0(%rsi),%ymm5
// CHECK-NEXT:  	vmovups 0xc0(%rsi),%ymm6
// CHECK-NEXT:  	vmovups 0xe0(%rsi),%ymm7
// CHECK-NEXT:  	mov    $0xffffffffffffffff,%rax
// CHECK-NEXT:  	xor    %ecx,%ecx
// CHECK-NEXT:  	vbroadcastss (%rdi,%rcx,1),%ymm8
// CHECK-NEXT:  	vfmadd213ps (%rdx,%rcx,1),%ymm0,%ymm8
// CHECK-NEXT:  	vbroadcastss 0x4(%rdi,%rcx,1),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm1,%ymm9
// CHECK-NEXT:  	vbroadcastss 0x8(%rdi,%rcx,1),%ymm8
// CHECK-NEXT:  	vfmadd213ps %ymm9,%ymm2,%ymm8
// CHECK-NEXT:  	vbroadcastss 0xc(%rdi,%rcx,1),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm3,%ymm9
// CHECK-NEXT:  	vbroadcastss 0x10(%rdi,%rcx,1),%ymm8
// CHECK-NEXT:  	vfmadd213ps %ymm9,%ymm4,%ymm8
// CHECK-NEXT:  	vbroadcastss 0x14(%rdi,%rcx,1),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm5,%ymm9
// CHECK-NEXT:  	vbroadcastss 0x18(%rdi,%rcx,1),%ymm8
// CHECK-NEXT:  	vfmadd213ps %ymm9,%ymm6,%ymm8
// CHECK-NEXT:  	vbroadcastss 0x1c(%rdi,%rcx,1),%ymm9
// CHECK-NEXT:  	vfmadd213ps %ymm8,%ymm7,%ymm9
// CHECK-NEXT:  	vmovups %ymm9,(%rdx,%rcx,1)
// CHECK-NEXT:  	inc    %rax
// CHECK-NEXT:  	add    $0x20,%rcx
// CHECK-NEXT:  	cmp    $0x7,%rax
// CHECK-NEXT:  	jb     <myfun+0x40>
// CHECK-NEXT:  	vzeroupper 
