# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_iree_runtime

import tempfile

import xtc.graphs.xtc.op as O
from xtc.backends.iree import Backend

I, J, K, dtype = 64, 64, 512, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

impl = Backend(gb.graph)

sch = impl.get_scheduler()
sch.tile("i", {"i1": 32, "i2": 8})
sch.tile("j", {"j1": 32, "j2": 8, "j3": 4})
sch.tile("k", {"k1": 16})
sch.parallelize(["i", "j"])
sch.vectorize(["j3"])

with tempfile.TemporaryDirectory() as tmp:
    comp = impl.get_compiler(
        dump_file=f"{tmp}/matmul_iree",
        print_source_ir=True,
        print_transformed_ir=True,
    )
    module = comp.compile(sch.schedule())
    res = module.get_executor(validate=True).execute()
print(f"CODE: {res}")

# CHECK:       // -----// IREE input MLIR //----- //
# CHECK-NEXT:  builtin.module {
# CHECK-NEXT:    func.func @matmul(%0 : tensor<64x512xf32>, %1 : tensor<512x64xf32>) -> tensor<64x64xf32> {
# CHECK-NEXT:      %2 = tensor.empty() : tensor<64x64xf32>
# CHECK-NEXT:      %3 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      %4 = linalg.fill {__xtc_id_C_0_} ins(%3 : f32) outs(%2 : tensor<64x64xf32>) -> tensor<64x64xf32>
# CHECK-NEXT:      %5 = linalg.matmul {__xtc_id_C_, compilation_info = #iree_codegen.compilation_info<
# CHECK-SAME:        lowering_config = #iree_cpu.lowering_config<
# CHECK-SAME:          distribution = [32, 32, 0]
# CHECK-SAME:          cache_parallel = [8, 8, 0]
# CHECK-SAME:          cache_reduction = [0, 0, 16]
# CHECK-SAME:          vector_common_parallel = [0, 4, 0]
# CHECK-SAME:          vector_reduction = [0, 0, 1]
# CHECK-SAME:        translation_info = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
# CHECK-SAME:      ins(%0, %1 : tensor<64x512xf32>, tensor<512x64xf32>) outs(%4 : tensor<64x64xf32>) -> tensor<64x64xf32>
# CHECK-NEXT:      func.return %5 : tensor<64x64xf32>
# CHECK-NEXT:    }
# CHECK-NEXT:  }

# CHECK:       // -----// IR Dump After GenericVectorizationPass (iree-codegen-generic-vectorization) //----- //
# CHECK-NEXT:  func.func @matmul_dispatch_0_matmul_64x64x512_f32() attributes {translation_info = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>} {
# CHECK-NEXT:    %cst = arith.constant dense<0.000000e+00> : vector<8x8xf32>
# CHECK-NEXT:    %0 = ub.poison : f32
# CHECK-NEXT:    %c4 = arith.constant 4 : index
# CHECK-NEXT:    %c8 = arith.constant 8 : index
# CHECK-NEXT:    %c32 = arith.constant 32 : index
# CHECK-NEXT:    %c1 = arith.constant 1 : index
# CHECK-NEXT:    %c16 = arith.constant 16 : index
# CHECK-NEXT:    %c512 = arith.constant 512 : index
# CHECK-NEXT:    %c0 = arith.constant 0 : index
# CHECK-NEXT:    %1 = hal.interface.binding.subspan {{.*}} binding(0) {{.*}} : memref<64x512xf32, #hal.descriptor_type<storage_buffer>>
# CHECK-NEXT:    %2 = hal.interface.binding.subspan {{.*}} binding(1) {{.*}} : memref<512x64xf32, #hal.descriptor_type<storage_buffer>>
# CHECK-NEXT:    %3 = hal.interface.binding.subspan {{.*}} binding(2) {{.*}} : memref<64x64xf32, #hal.descriptor_type<storage_buffer>>
# CHECK-NEXT:    %4 = iree_codegen.load_from_buffer %1 : memref<64x512xf32, #hal.descriptor_type<storage_buffer>> -> tensor<64x512xf32>
# CHECK-NEXT:    %5 = iree_codegen.load_from_buffer %2 : memref<512x64xf32, #hal.descriptor_type<storage_buffer>> -> tensor<512x64xf32>
# CHECK-NEXT:    %6 = tensor.empty() : tensor<64x64xf32>
# CHECK-NEXT:    %7 = scf.forall (%arg0, %arg1) = (0, 0) to (64, 64) step (32, 32) shared_outs(%arg2 = %6) -> (tensor<64x64xf32>) {
# CHECK-NEXT:      %extracted_slice = tensor.extract_slice %arg2[%arg0, %arg1] [32, 32] [1, 1] : tensor<64x64xf32> to tensor<32x32xf32>
# CHECK-NEXT:      %8 = scf.for %arg3 = %c0 to %c32 step %c8 iter_args(%arg4 = %extracted_slice) -> (tensor<32x32xf32>) {
# CHECK-NEXT:        %9 = scf.for %arg5 = %c0 to %c32 step %c8 iter_args(%arg6 = %arg4) -> (tensor<32x32xf32>) {
# CHECK-NEXT:          %extracted_slice_0 = tensor.extract_slice %arg6[%arg3, %arg5] [8, 8] [1, 1] : tensor<32x32xf32> to tensor<8x8xf32>
# CHECK-NEXT:          %10 = vector.transfer_write %cst, %extracted_slice_0[%c0, %c0] {in_bounds = [true, true]} : vector<8x8xf32>, tensor<8x8xf32>
# CHECK-NEXT:          %11 = scf.for %arg7 = %c0 to %c512 step %c16 iter_args(%arg8 = %10) -> (tensor<8x8xf32>) {
# CHECK-NEXT:            %12 = scf.for %arg9 = %c0 to %c8 step %c4 iter_args(%arg10 = %arg8) -> (tensor<8x8xf32>) {
# CHECK-NEXT:              %extracted_slice_1 = tensor.extract_slice %arg10[0, %arg9] [8, 4] [1, 1] : tensor<8x8xf32> to tensor<8x4xf32>
# CHECK-NEXT:              %13 = scf.for %arg11 = %c0 to %c16 step %c1 iter_args(%arg12 = %extracted_slice_1) -> (tensor<8x4xf32>) {
# CHECK-NEXT:                %14 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg3, %arg0]
# CHECK-NEXT:                %15 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg11, %arg7]
# CHECK-NEXT:                %extracted_slice_3 = tensor.extract_slice %4[%14, %15] [8, 1] [1, 1] : tensor<64x512xf32> to tensor<8x1xf32>
# CHECK-NEXT:                %16 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg11, %arg7]
# CHECK-NEXT:                %17 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>()[%arg9, %arg5, %arg1]
# CHECK-NEXT:                %extracted_slice_4 = tensor.extract_slice %5[%16, %17] [1, 4] [1, 1] : tensor<512x64xf32> to tensor<1x4xf32>
# CHECK-NEXT:                %18 = vector.transfer_read %extracted_slice_3[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<8x1xf32>, vector<8x1xf32>
# CHECK-NEXT:                %19 = vector.transfer_read %extracted_slice_4[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<1x4xf32>, vector<1x4xf32>
# CHECK-NEXT:                %20 = vector.transfer_read %arg12[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<8x4xf32>, vector<8x4xf32>
# CHECK-NEXT:                %21 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %18, %19, %20 : vector<8x1xf32>, vector<1x4xf32> into vector<8x4xf32>
# CHECK-NEXT:                %22 = vector.transfer_write %21, %arg12[%c0, %c0] {in_bounds = [true, true]} : vector<8x4xf32>, tensor<8x4xf32>
# CHECK-NEXT:                scf.yield %22 : tensor<8x4xf32>
# CHECK-NEXT:              }
# CHECK-NEXT:              %inserted_slice_2 = tensor.insert_slice %13 into %arg10[0, %arg9] [8, 4] [1, 1] : tensor<8x4xf32> into tensor<8x8xf32>
# CHECK-NEXT:              scf.yield %inserted_slice_2 : tensor<8x8xf32>
# CHECK-NEXT:            }
# CHECK-NEXT:            scf.yield %12 : tensor<8x8xf32>
# CHECK-NEXT:          }
# CHECK-NEXT:          %inserted_slice = tensor.insert_slice %11 into %arg6[%arg3, %arg5] [8, 8] [1, 1] : tensor<8x8xf32> into tensor<32x32xf32>
# CHECK-NEXT:          scf.yield %inserted_slice : tensor<32x32xf32>
# CHECK-NEXT:        }
# CHECK-NEXT:        scf.yield %9 : tensor<32x32xf32>
# CHECK-NEXT:      }
# CHECK-NEXT:      scf.forall.in_parallel {
# CHECK-NEXT:        tensor.parallel_insert_slice %8 into %arg2[%arg0, %arg1] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<64x64xf32>
# CHECK-NEXT:      }
# CHECK-NEXT:    } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
# CHECK-NEXT:    iree_codegen.store_to_buffer %7, %3 : tensor<64x64xf32> into memref<64x64xf32, #hal.descriptor_type<storage_buffer>>
# CHECK-NEXT:    return
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
