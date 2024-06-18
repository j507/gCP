#ifndef INCLUDE_LINE_SEARCH_H_
#define INCLUDE_LINE_SEARCH_H_

#include <gCP/run_time_parameters.h>

#include <deal.II/base/table_handler.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

namespace gCP
{

  /*!
   * @brief Line search algorithmus
   * @details The algorithmus mirrors that shown in chapter 6.5
   * "Global methods for systems of nonlinear equations" of
   * https://doi.org/10.1137/1.9781611971200
   *
   * @todo Docu
   */
  class LineSearch
  {
  public:
    LineSearch();

    LineSearch(const RunTimeParameters::LineSearchParameters &parameters);

    void reinit(const double initial_scalar_function_value,
                const unsigned int step_id = 0,
                const unsigned int iteration_id = 0);

    void reinit(const std::vector<double> initial_scalar_function_values,
                const unsigned int step_id = 0,
                const unsigned int iteration_id = 0);

    bool suficient_descent_condition(
        const double trial_scalar_function_value,
        const double lambda) const;

    bool suficient_descent_condition(
        const std::vector<double> trial_scalar_function_values,
        const std::vector<double> lambdas);

    double get_lambda(const double trial_scalar_function_value,
                      const double current_lambda);

    std::vector<double> get_lambdas(
        const std::vector<double> trial_scalar_function_values,
        const std::vector<double> current_lambdas);

    unsigned int get_n_iterations() const;

    static double get_objective_function_value(const double residuum);

    static std::vector<double> get_objective_function_values(
        const std::vector<double> residuum);

    void write_to_file(const std::string filepath);

  private:
    unsigned int n_iterations;

    const unsigned int n_max_iterations;

    const double alpha;

    const double beta;

    double initial_scalar_function_value;

    std::vector<double> initial_scalar_function_values;

    double descent_direction;

    std::vector<double> descent_directions;

    double lambda;

    std::vector<double> lambdas;

    double old_lambda;

    std::vector<double> old_lambdas;

    double old_old_lambda;

    std::vector<double> old_old_lambdas;

    double old_scalar_function_value;

    std::vector<double> old_scalar_function_values;

    double old_old_scalar_function_value;

    std::vector<double> old_old_scalar_function_values;

    std::vector<bool> flag_sufficient_descent_conditions;

    unsigned int step_id;

    unsigned int iteration_id;

    dealii::TableHandler table_handler;

    double quadratic_backtracking(
        const double trial_scalar_function_value);

    double quadratic_backtracking(
        const double trial_scalar_function_value,
        const unsigned int id);

    double cubic_backtracking();

    double cubic_backtracking(const unsigned int id);
  };

  inline bool
  LineSearch::suficient_descent_condition(
      const double trial_scalar_function_value,
      const double lambda) const
  {
    if (n_iterations > n_max_iterations)
    {
      std::cout
          << "Warning: The maximal number of iterations of the line search"
          << " algorithm has been reached" << std::endl;

      return true;
    }
    else
    {
      /*const bool beta_condition =
          trial_scalar_function_value > initial_scalar_function_value +
                                            beta * lambda * descent_direction;

      if (!beta_condition)
      {
        std::cout << "Second conditions not fulfilled" << std::endl;
      }*/

      return (trial_scalar_function_value <
              initial_scalar_function_value +
                  alpha * lambda * descent_direction);
    }
  }

  inline bool
  LineSearch::suficient_descent_condition(
      const std::vector<double> trial_scalar_function_values,
      const std::vector<double> lambdas)
  {
    if (n_iterations > n_max_iterations)
    {
      std::cout
          << "Warning: The maximal number of iterations of the line search"
          << " algorithm has been reached" << std::endl;

      return true;
    }
    else
    {
      /*const bool beta_condition =
        trial_scalar_function_value > initial_scalar_function_value +
          beta*lambda*descent_direction;

      if (!beta_condition)
      {
        std::cout << "Second conditions not fulfilled" << std::endl;
      }*/

      for (unsigned int i = 0;
           i < flag_sufficient_descent_conditions.size();
           ++i)
      {
        flag_sufficient_descent_conditions[i] =
            trial_scalar_function_values[i] <
            initial_scalar_function_values[i] +
                alpha * lambdas[i] * descent_directions[i];


        if (trial_scalar_function_values[i] == 0.0)
        {
          flag_sufficient_descent_conditions[i] = true;
        }
      }

      return (flag_sufficient_descent_conditions[0] &&
              flag_sufficient_descent_conditions[1]);
    }
  }

  inline double
  LineSearch::get_objective_function_value(const double residuum)
  {
    return (0.5 * residuum * residuum);
  }

  inline std::vector<double>
  LineSearch::get_objective_function_values(const std::vector<double> residuum)
  {
    std::vector<double> objective_functions(residuum.size());

    for (unsigned int i = 0; i < residuum.size(); i++)
    {
      objective_functions[i] = 0.5 * residuum[i] * residuum[i];
    }

    return (objective_functions);
  }

  inline unsigned int
  LineSearch::get_n_iterations() const
  {
    return (n_iterations);
  }

} // namespace LineSearch

#endif /* INCLUDE_LINE_SEARCH_H_ */