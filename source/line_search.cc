#include <gCP/line_search.h>


namespace gCP
{



LineSearch::LineSearch()
:
n_iterations(0),
n_max_iterations(15),
alpha(1e-4),
beta(0.9)
{
  table_handler.declare_column("S");
  table_handler.declare_column("N-Itr");
  table_handler.declare_column("L-Itr");
  table_handler.declare_column("Rlx-Prm");
  table_handler.set_scientific("Rlx-Prm", true);
}



LineSearch::LineSearch(
  const RunTimeParameters::LineSearchParameters &parameters)
:
n_iterations(0),
n_max_iterations(parameters.n_max_iterations),
alpha(parameters.alpha),
beta(parameters.beta)
{
  table_handler.declare_column("S");
  table_handler.declare_column("N-Itr");
  table_handler.declare_column("L-Itr");
  table_handler.declare_column("Rlx-Prm");
  table_handler.set_scientific("Rlx-Prm", true);
}



void LineSearch::reinit(
  const double        initial_scalar_function_value,
  const unsigned int  step_id,
  const unsigned int  iteration_id)
{
  this->n_iterations = 0;

  this->initial_scalar_function_value = initial_scalar_function_value;

  this->descent_direction = -2. * initial_scalar_function_value;

  this->step_id = step_id;

  this->iteration_id = iteration_id;
}



double LineSearch::get_lambda(const double trial_scalar_function_value,
                              const double trial_lambda)
{
  AssertIsFinite(trial_scalar_function_value);
  AssertIsFinite(trial_lambda);

  old_old_scalar_function_value = old_scalar_function_value;
  old_scalar_function_value     = trial_scalar_function_value;

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

  table_handler.add_value("S", step_id);
  table_handler.add_value("N-Itr", iteration_id);
  table_handler.add_value("L-Itr", n_iterations);
  table_handler.add_value("Rlx-Prm", lambda);

  return lambda;
}



void LineSearch::write_to_file(const std::string filepath)
{
  std::ofstream file(filepath);

  table_handler.write_text(
    file,
    dealii::TableHandler::TextOutputFormat::org_mode_table);

  file.close();
}



double LineSearch::quadratic_backtracking(
  const double trial_scalar_function_value)
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