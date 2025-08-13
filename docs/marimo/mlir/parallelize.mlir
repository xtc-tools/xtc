func.func @matmul(%A: memref<4x8xf64>, %B: memref<8x16xf64>, %C: memref<4x16xf64>){
	linalg.matmul {
		loop.dims = ["i", "j", "k"],
		loop.schedule = {
			"i" = {"parallelize"},
				"k",
					"j"
		}
	}
	ins(%A, %B : memref<4x8xf64>, memref<8x16xf64>)
	outs(%C: memref<4x16xf64>)
	return
}