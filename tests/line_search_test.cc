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

  dealii::Vector<double>      initial_residual;

  dealii::Vector<double>      newton_direction;

  double                      initial_scalar_function_value;

  double                      trial_scalar_function_value;

  gCP::LineSearch             line_search;

  dealii::Vector<double> get_residual(dealii::Vector<double>  &x) const;
};



LineSearch::LineSearch()
:
pcout(std::cout,
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
{
  initial_guess.reinit(2);
  initial_residual.reinit(2);
  newton_direction.reinit(2);

  initial_guess[0] = 2.0;
  initial_guess[1] = 0.5;

  newton_direction[0] = -3.;
  newton_direction[1] = 9.74;

  initial_residual = get_residual(initial_guess);

  initial_scalar_function_value =
    0.5 * std::pow(initial_residual.l2_norm(),2);
}



void LineSearch::run()
{
  double lambda = 1.;

  line_search.reinit(initial_scalar_function_value);

  dealii::Vector<double> trial_solution(initial_guess);

  trial_solution = initial_guess;
  trial_solution.add(lambda, newton_direction);

  dealii::Vector<double> trial_residual = get_residual(trial_solution);

  trial_scalar_function_value = 0.5 * std::pow(trial_residual.l2_norm(),2);

  while (!line_search.suficient_descent_condition(trial_scalar_function_value,lambda))
  {
    lambda =  line_search.get_lambda(trial_scalar_function_value,
                                     lambda);

    std::cout << "lambda = " << lambda << std::endl;

    trial_solution = initial_guess;
    trial_solution.add(lambda, newton_direction);
    trial_residual = get_residual(trial_solution);
    trial_scalar_function_value =
      0.5 * std::pow(trial_residual.l2_norm(),2);
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