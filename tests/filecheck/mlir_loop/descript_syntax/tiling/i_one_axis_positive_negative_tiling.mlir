// RUN:not mlir-loop --no-alias --print-source-ir %s 2>&1 | filecheck %s
func.func @matmul(%A: memref<256x512xf64>, %B: memref<512x256xf64>, %C: memref<256x256xf64>){
   linalg.matmul {
         loop.dims = ["i", "j", "k"],
         loop.schedule = {
            "i",
              "k",
                "k#32",
                  "k#-1",
                     "j"
         }
      }
      ins(%A, %B : memref<256x512xf64>, memref<512x256xf64>)
      outs(%C: memref<256x256xf64>)
      return
 }
// CHECK: xtc.schedules.descript.ScheduleInterpretError: `k#-1`: -1 is not an integer.
