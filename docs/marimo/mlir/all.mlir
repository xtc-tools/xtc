func.func @matmul(%A: memref<256x512xf64>, %B: memref<512x256xf64>, %C: memref<256x256xf64>){
	linalg.matmul {
		loop.dims = ["i", "j", "k"],
		loop.schedule = {
			"i",
				"j",
					"k",
						"i#1" = {"unroll"},
							"k#8" = {"unroll"},
								"j#32" = {"vectorize"}
		}
	}
	ins(%A, %B : memref<256x512xf64>, memref<512x256xf64>)
	outs(%C: memref<256x256xf64>)
	return
}