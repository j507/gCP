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

    bool suficient_descent_condition(
        const double trial_scalar_function_value,
        const double lambda) const;

    double get_lambda(const double trial_scalar_function_value,
                      const double current_lambda);

    unsigned int get_n_iterations() const;

    static double get_objective_function_value(const double residuum);

    void write_to_file(const std::string filepath);

  private:
    unsigned int n_iterations;

    const unsigned int n_max_iterations;

    const double alpha;

    const double beta;

    double initial_scalar_function_value;

    double descent_direction;

    double lambda;

    double old_lambda;

    double old_old_lambda;

    double old_scalar_function_value;

    double old_old_scalar_function_value;

    unsigned int step_id;

    unsigned int iteration_id;

    dealii::TableHandler table_handler;

    double quadratic_backtracking(
        const double trial_scalar_function_value);

    double cubic_backtracking();
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



  inline double
  LineSearch::get_objective_function_value(const double residuum)
  {
    return (0.5 * residuum * residuum);
  }



  inline unsigned int
  LineSearch::get_n_iterations() const
  {
    return (n_iterations);
  }

} // namespace LineSearch

#endif /* INCLUDE_LINE_SEARCH_H_ */