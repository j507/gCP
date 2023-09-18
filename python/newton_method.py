import math
import numpy
import matplotlib.pyplot

def sig(x, h):
    return math.erf(math.sqrt(math.pi)/2 * x / h)

def dsig(x, h):
    return 1 / h * math.exp(-math.pi * x * x / h / h / 4)

def ddsig(x, h):
    return -0.5 * math.pi * x / math.pow(h,3) * math.exp(-math.pi * x * x / h / h / 4)

def get_residual(x, h, f):
    return (f - sig(x,h))

def get_jacobian(x, h):
    return (-dsig(x,h))

def get_second_jacobian(x,h):
    return (-ddsig(x,h))

regularization_parameter  = 1e-9
solution                  = 1e-3
value_at_solution         = sig(solution, regularization_parameter)
initial_guess             = 0
constant                  = 1e-10
trial_solution            = initial_guess
convergence               = False
iteration                 = 0
max_n_iteration           = 50
old_residual              = 0
absolute_tolerance        = 1e-8
solution_history = numpy.array([trial_solution])

fig, axs = matplotlib.pyplot.subplots(2, 2)

residual = get_residual(trial_solution,
                        regularization_parameter,
                        value_at_solution)

print(f"r({int(iteration)}) = {residual:e}")

while (not convergence):
  iteration = iteration + 1

  old_residual = residual

  residual = get_residual(trial_solution,
                          regularization_parameter,
                          value_at_solution)

  jacobian = get_jacobian(trial_solution, regularization_parameter)

  step = residual / jacobian

  trial_solution = trial_solution - step

  solution_history = numpy.append(solution_history, [trial_solution])

  residual = get_residual(trial_solution,
                          regularization_parameter,
                          value_at_solution)

  convergence_rate =  0.

  if (iteration > 1):
    convergence_rate = math.log(residual) / math.log(old_residual)

  print(f"x_n({int(iteration)}) = {step:e}, r({int(iteration)}) = {residual:e}, C-Rate = {convergence_rate}")

  if (residual < absolute_tolerance):
      print("Absolute tolerance reached")
      convergence = True

  if (iteration == max_n_iteration):
      print("Maximum number of iterations reached")
      break

value_history = \
  numpy.array([sig(i, regularization_parameter) \
                for i in solution_history])

axs[0,0].plot(solution_history, value_history, 'bs')

constant                  = 1e-10
trial_solution            = initial_guess
old_trial_solution        = initial_guess
convergence               = False
iteration                 = 0
old_residual              = 0
solution_history = numpy.array([trial_solution])

residual = get_residual(trial_solution,
                        regularization_parameter,
                        value_at_solution)

print(f"r({int(iteration)}) = {residual:e}")

while (not convergence):
  iteration = iteration + 1

  old_residual            = residual
  old_old_trial_solution  = old_trial_solution
  old_trial_solution      = trial_solution

  if (iteration > 2):
     constant = old_old_trial_solution


  residual = get_residual(trial_solution,
                          regularization_parameter,
                          value_at_solution)

  residual_c = get_residual(constant,
                            regularization_parameter,
                            value_at_solution)

  jacobian = get_jacobian(trial_solution, regularization_parameter)

  trial_solution = trial_solution - \
                   (trial_solution - constant) * residual / \
                   (residual - (trial_solution - constant) * jacobian * \
                   residual_c / (residual - residual_c))

  solution_history = numpy.append(solution_history, [trial_solution])

  residual = get_residual(trial_solution,
                          regularization_parameter,
                          value_at_solution)

  convergence_rate =  0.

  if (iteration > 1):
    convergence_rate = math.log(residual) / math.log(old_residual)

  print(f"r({int(iteration)}) = {residual:e}, C-Rate = {convergence_rate}")

  if (residual < absolute_tolerance):
      print("Absolute tolerance reached")
      convergence = True

  if (iteration == max_n_iteration):
      print("Maximum number of iterations reached")
      break

value_history = \
  numpy.array([sig(i, regularization_parameter) \
                for i in solution_history])

axs[0,1].plot(solution_history, value_history, 'bs')

trial_solution            = initial_guess
old_trial_solution        = initial_guess
convergence               = False
iteration                 = 0
old_residual              = 0
solution_history = numpy.array([trial_solution])

residual = get_residual(trial_solution,
                        regularization_parameter,
                        value_at_solution)

print(f"r({int(iteration)}) = {residual:e}")

while (not convergence):
  iteration = iteration + 1

  old_residual            = residual
  old_trial_solution      = trial_solution

  residual = get_residual(trial_solution,
                          regularization_parameter,
                          value_at_solution)

  jacobian = get_jacobian(trial_solution, regularization_parameter)

  second_jacobian = get_second_jacobian(trial_solution, regularization_parameter)

  trial_solution = trial_solution - \
                   (residual / jacobian) /  \
                   (1 - residual * second_jacobian / 2 / jacobian / jacobian)

  solution_history = numpy.append(solution_history, [trial_solution])

  residual = get_residual(trial_solution,
                          regularization_parameter,
                          value_at_solution)

  convergence_rate =  0.

  if (iteration > 1):
    convergence_rate = math.log(residual) / math.log(old_residual)

  print(f"r({int(iteration)}) = {residual:e}, C-Rate = {convergence_rate}")

  if (residual < absolute_tolerance):
      print("Absolute tolerance reached")
      convergence = True

  if (iteration == max_n_iteration):
      print("Maximum number of iterations reached")
      break

value_history = \
  numpy.array([sig(i, regularization_parameter) \
                for i in solution_history])

axs[1,0].plot(solution_history, value_history, 'bs')

matplotlib.pyplot.show()
