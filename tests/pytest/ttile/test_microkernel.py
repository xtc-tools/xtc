from pathlib import Path

from xtc.schedules.ttile.computation import Computation, Computation_spec
from xtc.schedules.ttile.archi import pinocchio_machine, laptop_guillaume_machine

from xtc.schedules.ttile.microkernel import generate_microkernels, launch_microkernel_testing
from xtc.schedules.ttile.microkernel import load_microkernel_info, sort_microkernels_by_pperf


# Note: we skip microkernel executions (because this is too long for unit test)


# Test of the microkernel candidate generation for convolutions
def test_microkernel_gen_conv():
	machine = laptop_guillaume_machine
	comp = Computation(Computation_spec.CONV, 4)			# Float32, 4 octets

	lmickern = generate_microkernels(machine, comp)

	assert(len(lmickern)==1007)
	assert(str(lmickern[-1]) == "(conv2d(f32); Architype_enum.INTEL, 16 vreg) [V (f, 8); U(f, 9)]")

	return


# Test of the microkernel candidate generation for matmult
def test_microkernel_gen_matmult():
	machine = pinocchio_machine
	comp = Computation(Computation_spec.MATMULT, 4)

	lmickern = generate_microkernels(machine, comp)

	assert(len(lmickern)==59)
	assert(str(lmickern[-1]) == "(matmul(f32); Architype_enum.INTEL, 32 vreg) [V (j, 16); U(i, 19)]")

	return


# Test of the microkernel generation and measurement for matmult
# WARNING - actual execution on a machine. This was done on "laptop_guillaume_machine"
def test_launch_microkernel_testing_matmult():
	machine = laptop_guillaume_machine
	comp = Computation(Computation_spec.MATMULT, 4)

	lmickern = generate_microkernels(machine, comp)
	assert(len(lmickern) == 30)

	backend = "tvm"
	peak_perf = launch_microkernel_testing(machine, comp, lmickern[9], backend)
	assert(peak_perf != None)

	return


# Test of the microkernel file save mechanism
def test_run_and_save_microkernel_info():
	# Note: that config was chosen since there are only few microkernels.
	machine = pinocchio_machine
	comp = Computation(Computation_spec.CONV, 4)

	name_outfile = str(Path(__file__).parent / "mickern_xtctvm_pinocchio_conv_f32.csv")
	(machine_name, comp_name, ld_mickern_info) = load_microkernel_info(name_outfile)

	# Sort it
	ld_mickern_info = sort_microkernels_by_pperf(ld_mickern_info)

	assert(machine_name == machine.name)
	assert(str(comp) == comp_name)
	assert(str(ld_mickern_info[-1]) == "{'f': 2, 'x': 1, 'y': 12, 'c': 5, 'h': 1, 'w': 1, 'peak_perf': 99.048}")

	return


