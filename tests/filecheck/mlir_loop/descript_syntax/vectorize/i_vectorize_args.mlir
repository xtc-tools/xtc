// RUN: not mlir-loop --no-alias --print-source-ir %s 2>&1 | filecheck %s

func.func @matmul(%A: memref<256x512xf64>, %B: memref<512x256xf64>, %C: memref<256x256xf64>){
	linalg.matmul {
		loop.dims = ["i", "j", "k"],
		loop.schedule = {
			"i" = {"parallelize"},
				"j",
					"k",
						"i#1" = {"unroll"},
							"k#8" = {"unroll"},
								"j#32" = {"vectorize" = 1}
		}
	}
	ins(%A, %B: memref<256x512xf64>, memref<512x256xf64>)
	outs(%C: memref<256x256xf64>)
	return
}

// CHECK: `{"vectorize" = 1}`: vectorization parameter should be True, False or a string.
