#include <gCP/line_search.h>


namespace gCP
{



LineSearch::LineSearch()
:
n_iterations(0),
alpha(1e-4)
{}



void LineSearch::reinit(
  const double initial_scalar_function_value)
{
  this->n_iterations                  = 0;
  this->initial_scalar_function_value = initial_scalar_function_value;
  this->descent_direction             = - 2. * initial_scalar_function_value;
}



double LineSearch::get_lambda(const double trial_scalar_function_value,
                              const double trial_lambda)
{
  old_old_scalar_function_value = old_scalar_function_value;
  old_scalar_function_value     = trial_scalar_function_value;

  double lambda;

  if (n_iterations == 0)
  {
    old_lambda  = trial_lambda;
    lambda      = quadratic_backtracking(trial_scalar_function_value);
  }
  else
  {
    old_old_lambda  = old_lambda;
    old_lambda      = trial_lambda;
    lambda          = cubic_backtracking();
  }

  n_iterations++;

  AssertIsFinite(lambda);

  return lambda;
}



double LineSearch::quadratic_backtracking(const double trial_scalar_function_value)
{
  double lambda = - descent_direction / 2.0 /
                  (trial_scalar_function_value -
                   initial_scalar_function_value -
                   descent_direction);

  if (lambda < 0.1)
    return 0.1;
  else if (lambda > 0.5)
    return 0.5;
  else
    return lambda;
}



double LineSearch::cubic_backtracking()
{
  const double factor = 1./(old_lambda - old_old_lambda);

  dealii::FullMatrix<double> matrix;
  matrix.reinit(2,2);
  matrix[0][0] = 1./(old_lambda * old_lambda);
  matrix[0][1] = - 1./(old_old_lambda * old_old_lambda);
  matrix[1][0] = -old_old_lambda/(old_lambda * old_lambda);;
  matrix[1][1] = old_lambda/(old_old_lambda * old_old_lambda);

  dealii::Vector<double> vector;
  vector.reinit(2);
  vector[0] = old_scalar_function_value -
              initial_scalar_function_value -
              old_lambda * descent_direction;
  vector[1] = old_old_scalar_function_value -
              initial_scalar_function_value -
              old_old_lambda * descent_direction;

  dealii::Vector<double> coefficients;
  coefficients.reinit(2);

  matrix.vmult(coefficients, vector);

  coefficients *= factor;

  double lambda = (-coefficients[1] +
                  std::sqrt(coefficients[1]*coefficients[1] -
                            3.0 * coefficients[0] *descent_direction)) /
                  (3. * coefficients[0]);

  if (lambda < 0.1*old_lambda)
    return 0.1*old_lambda;
  else if (lambda > 0.5*old_lambda)
    return 0.5*old_lambda;
  else
    return lambda;
}



} // namespace gCP