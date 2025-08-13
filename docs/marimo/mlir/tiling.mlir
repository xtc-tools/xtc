func.func @matmul(%A: memref<4x8xf64>, %B: memref<8x16xf64>, %C: memref<4x16xf64>){
	linalg.matmul {
		loop.dims = ["i", "j", "k"],
		loop.schedule = {
			"i",
				"j",
					"k",
						"i#1",
							"j#8",
								"k#8"
		}
	}
	ins(%A, %B : memref<4x8xf64>, memref<8x16xf64>)
	outs(%C: memref<4x16xf64>)
	return
}