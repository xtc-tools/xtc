import logging
from pathlib import Path
import itertools
import subprocess
import argparse

from xtc.utils.text import jinja_generate_file

logger = logging.getLogger(__name__)

THIS_DIR = Path(__file__).parent

def compile_csrc(input: str):
  output = Path(input).with_suffix(".o")
  logger.info(f"Compiling {input} -> {output}")
  subprocess.run(f"gcc -O2 -march=native -mtune=native {input} -o {output} -c", text=True, shell=True, check=True)

def test_csrc(input: str):
  output = Path(input).with_suffix(".test")
  logger.info(f"Compiling test {input} -> {output}")
  subprocess.run(f"gcc -O2 -DTEST -march=native -mtune=native {input} -o {output}", text=True, shell=True, check=True)
  logger.info(f"Running test {output}")
  subprocess.run(f"./{output}")

def generate_all_uk(dest_fmt: str, template: str, all_i: list[int], all_j: list[int], compile: bool = False, test: bool = False):
  uks = itertools.product(all_i, all_j)
  for i, j in uks:
    dest = generate_uk_ixj(dest_fmt, template, i=i, j=j)
    if compile:
      compile_csrc(dest)
    if test:
      test_csrc(dest)
  
def generate_uk_ixj(dest_fmt: str, template: str, i: int, j: int) -> str:
  dest = dest_fmt.format(i=i, j=j)
  logger.info(f"Generating uk {i}x{j} into {dest}")
  jinja_generate_file(dest, str(template), I=i, J=j)
  return dest

def main():
  this_dir = Path(__file__).parent
  default = argparse.Namespace(
    ii = [1, 2, 3, 4, 5, 6, 7, 8],
    jj = [1, 2, 4, 8, 16, 32, 64, 128],
    template = str(this_dir / "external_matmul_uk_ixjxk.c.jinja"),
    dest_fmt ="external_matmul_uk_ixjxk_{i}x{j}.c",
    compile=True,
    test=True,
  )
  parser = argparse.ArgumentParser("Generates micro kparserernels")
  parser.add_argument("--i", type=int, nargs="+", default=default.ii, help="list of i values")
  parser.add_argument("--j", type=int, nargs="+", default=default.jj, help="list of j values")
  parser.add_argument("--template", default=default.template, help="template file")
  parser.add_argument("--dest-fmt", default=default.dest_fmt, help="output format file")
  parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=default.compile, help="compile output")
  parser.add_argument("--test", action=argparse.BooleanOptionalAction, default=default.test, help="test")

  logging.basicConfig()
  logger.setLevel(logging.INFO)

  args = parser.parse_args()

  generate_all_uk(args.dest_fmt, args.template, args.i, args.j, compile=args.compile, test=args.test)

if __name__ == "__main__":
  main()
