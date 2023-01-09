#ifndef INCLUDE_LINE_SEARCH_H_
#define INCLUDE_LINE_SEARCH_H_

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

  void reinit(const double initial_scalar_function_value);

  bool suficient_descent_condition(
    const double trial_scalar_function_value,
    const double lambda) const;

  double get_lambda(const double trial_scalar_function_value,
                    const double current_lambda);

  unsigned int get_n_iterations() const;

private:
  unsigned int  n_iterations;

  const double  alpha;

  double        initial_scalar_function_value;

  double        descent_direction;

  double        old_lambda;

  double        old_old_lambda;

  double        old_scalar_function_value;

  double        old_old_scalar_function_value;

  double quadratic_backtracking(
    const double trial_scalar_function_value);

  double cubic_backtracking();
};



inline bool
LineSearch::suficient_descent_condition(
  const double trial_scalar_function_value,
  const double lambda) const
{
  return (trial_scalar_function_value <
            initial_scalar_function_value +
            alpha*lambda*descent_direction);
}



inline unsigned int
LineSearch::get_n_iterations() const
{
  return (n_iterations);
}



} // namespace LineSearch



#endif /* INCLUDE_LINE_SEARCH_H_ */