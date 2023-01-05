#include <gCP/line_search.h>

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/vector.h>

#include <cmath>

namespace Tests
{



class LineSearch
{
public:

  LineSearch();

  void run();

private:

  dealii::ConditionalOStream  pcout;

  dealii::Vector<double>      initial_guess;

  double                      initial_scalar_function_value;

  double                      trial_scalar_function_value;

  gCP::LineSearch             line_search;

  dealii::Vector<double> get_residual(dealii::Vector<double>  &x) const;

  dealii::FullMatrix<double> get_jacobian(dealii::Vector<double> &x) const;
};



LineSearch::LineSearch()
:
pcout(std::cout,
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
{
  initial_guess.reinit(2);

  initial_guess[0] = 2.0;
  initial_guess[1] = 0.5;
}



void LineSearch::run()
{
  for (unsigned int i = 0; i < 3; ++i)
  {
    std::cout << "Iteration " << (i+1) << std::endl;

    // Assemble and solve the linear system
    const dealii::Vector<double> residual = get_residual(initial_guess);

    double initial_scalar_function_value =
      0.5 * std::pow(residual.l2_norm(),2);

    dealii::FullMatrix<double> jacobian = get_jacobian(initial_guess);

    dealii::Vector<double> newton_direction(2);

    jacobian.gauss_jordan();

    jacobian.vmult(newton_direction, residual);

    newton_direction *= -1.0;

    dealii::Vector<double> trial_solution(initial_guess);

    trial_solution = initial_guess;

    double lambda = 1.;

    trial_solution.add(lambda, newton_direction);

    dealii::Vector<double> trial_residual = get_residual(trial_solution);

    trial_scalar_function_value = 0.5 * std::pow(trial_residual.l2_norm(),2);

    // Line search algorithm
    line_search.reinit(initial_scalar_function_value);

    while (!line_search.suficient_descent_condition(trial_scalar_function_value,lambda))
    {
      // Compute new lambda
      std::cout << "  Condition not met. Computing a new lambda.."
                << std::endl;

      lambda =
        line_search.get_lambda(trial_scalar_function_value, lambda);

      std::cout << "    lambda = " << lambda << std::endl;

      // Compute new trial solution
      trial_solution = initial_guess;

      trial_solution.add(lambda, newton_direction);

      // Get residual to check the suficient descent condition
      trial_residual = get_residual(trial_solution);

      std::cout << "    residual = " << trial_residual << std::endl;

      trial_scalar_function_value =
        0.5 * std::pow(trial_residual.l2_norm(),2);
    }

    initial_guess = trial_solution;

    std::cout << "line search iterations = "
              << line_search.get_n_iterations() << std::endl;
  }
}



dealii::Vector<double> LineSearch::get_residual(
  dealii::Vector<double>  &x) const
{
  dealii::Vector<double> residual;

  residual.reinit(2);

  residual[0] = x[0]*x[0] + x[1]*x[1] - 2.;
  residual[1] = std::exp(x[0]-1.0) + x[1]*x[1]*x[1] - 2.;

 return residual;
}



dealii::FullMatrix<double> LineSearch::get_jacobian(
  dealii::Vector<double>  &x) const
{
  dealii::FullMatrix<double> jacobian(2);

  jacobian[0][0] = 2.0*x[0];
  jacobian[0][1] = 2.0*x[1];
  jacobian[1][0] = std::exp(x[0]-1.0);
  jacobian[1][1] = 3.0*x[1]*x[1];

 return jacobian;
}



} // namespace Test




int main(int argc, char *argv[])
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc, argv, dealii::numbers::invalid_unsigned_int);

    Tests::LineSearch line_search;
    line_search.run();

  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  return 0;
}