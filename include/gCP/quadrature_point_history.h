#ifndef INCLUDE_POINT_HISTORY_H_
#define INCLUDE_POINT_HISTORY_H_

#include <gCP/run_time_parameters.h>



namespace gCP
{



template <int dim>
struct QuadraturePointHistory
{
public:
  QuadraturePointHistory();

  virtual ~QuadraturePointHistory() = default;

  double get_slip_resistance(const unsigned int slip_id);

  void init(
    const RunTimeParameters::ScalarMicroscopicStressLawParameters
      &parameters,
    const unsigned int n_slips);

  void update_values(
    const unsigned int                      q_point,
    const std::vector<std::vector<double>>  &slips,
    const std::vector<std::vector<double>>  &old_slips);

private:
  const unsigned int  n_slips;

  const double        initial_slip_resistance;

  std::vector<double> slip_resistances;

  const double        linear_hardening_modulus;

  const double        hardening_parameter;

  double get_hardening_matrix_entry(const bool self_hardening) const;
};



} // namespace gCP



#endif /* INCLUDE_POINT_HISTORY_H_ */
