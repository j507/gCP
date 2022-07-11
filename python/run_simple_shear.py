import numpy
import argparse     # Module to parse arguments
import subprocess   # Module to execute terminal commands
import os

def nproc_type(x):
  """nproc_type A function acting as a type for admissible number of
  processors, i.e. only positive integers bigger than zero

  :param x: Desired number of processors
  :type x: int
  :raises argparse.ArgumentTypeError: Error if `x` is smaller than 1
  :return: Admissible number of proccesors
  :rtype: int

  """
  x = int(x)
  if x < 1:
    raise argparse.ArgumentTypeError("Minimum number of processors is 1")
  return x

def main(nproc):

  os.chdir('..')
  print(os.getcwd())
  prm_files = ["input/simple_shear_Bittencourt/double_slip_hl_25_Ht_0.prm",
               "input/simple_shear_Bittencourt/double_slip_hl_25_Ht_2e-1.prm",
               "input/simple_shear_Bittencourt/double_slip_hl_25_Ht_2e-2.prm",
               "input/simple_shear_Bittencourt/double_slip_hl_25_Ht_2e0.prm",
               "input/simple_shear_Bittencourt/double_slip_hl_35e-1_Ht_2e0.prm",
               "input/simple_shear_Bittencourt/double_slip_hl_125e-2_Ht_0.prm",
               "input/simple_shear_Bittencourt/double_slip_hl_125e-2_Ht_2e-1.prm",
               "input/simple_shear_Bittencourt/double_slip_hl_125e-2_Ht_2e-2.prm",
               "input/simple_shear_Bittencourt/double_slip_hl_125e0_Ht_0.prm",
               "input/simple_shear_Bittencourt/double_slip_hl_125e0_Ht_2e-1.prm",
               "input/simple_shear_Bittencourt/double_slip_hl_125e0_Ht_2e-2.prm",
               "input/simple_shear_Bittencourt/single_slip_hl_125e-2_Ht_0.prm",
               "input/simple_shear_Bittencourt/single_slip_hl_125e0_Ht_0.prm",
               "input/simple_shear_Kergassner/double_slip_lh_1_H_0.prm",
               "input/simple_shear_Kergassner/double_slip_lh_2_H_0.prm",
               "input/simple_shear_Kergassner/double_slip_lh_2e-1_H_0.prm",
               "input/simple_shear_Kergassner/double_slip_lh_2e-1_H_100.prm",
               "input/simple_shear_Kergassner/double_slip_lh_2e-1_H_1000.prm"]

  for prm_file in prm_files:
    process = subprocess.run(["mpirun", "-np", str(nproc), "./simple_shear", prm_file])

  prm_files = ["input/simple_shear_Kergassner/grain_boundaries_lh_2e-1.prm",
               "input/simple_shear_Kergassner/grain_boundaries_lh_2e0_alpha_0.prm",
               "input/simple_shear_Kergassner/grain_boundaries_lh_2e0_alpha_20.prm",
               "input/simple_shear_Kergassner/grain_boundaries_lh_2e0.prm",
               "input/simple_shear_Kergassner/grain_boundaries_lh_5e0_lambda_1e4.prm"
               "input/simple_shear_Kergassner/grain_boundaries_lh_5e0_lambda_0.prm"]

  for prm_file in prm_files:
    process = subprocess.run(["./simple_shear", prm_file])

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--nproc",
                      "-np",
                      type = nproc_type,
                      default = 8,
                      help = "Number of processors")
  args = parser.parse_args()
  main(args.nproc)