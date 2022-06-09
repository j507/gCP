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

  double get_slip_resistance(const unsigned int slip_id) const;

  std::vector<double> get_slip_resistances() const;

  void init(
    const RunTimeParameters::ScalarMicroscopicStressLawParameters
      &parameters,
    const unsigned int n_slips);

  void update_values(
    const unsigned int                      q_point,
    const std::vector<std::vector<double>>  &slips,
    const std::vector<std::vector<double>>  &old_slips);

private:
  unsigned int  n_slips;

  double              initial_slip_resistance;

  std::vector<double> slip_resistances;

  double              linear_hardening_modulus;

  double              hardening_parameter;

  bool                flag_init_was_called;

  double get_hardening_matrix_entry(const bool self_hardening) const;
};



template <int dim>
inline double
QuadraturePointHistory<dim>::get_slip_resistance(
  const unsigned int slip_id) const
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The QuadraturePointHistory<dim> "
                                 "instance has not been initialized."));

  return (slip_resistances[slip_id]);
}



template <int dim>
inline std::vector<double>
QuadraturePointHistory<dim>::get_slip_resistances() const
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The QuadraturePointHistory<dim> "
                                 "instance has not been initialized."));

  return (slip_resistances);
}



template <int dim>
inline double
QuadraturePointHistory<dim>::get_hardening_matrix_entry(
  const bool self_hardening) const
{
  AssertThrow(flag_init_was_called,
              dealii::ExcMessage("The QuadraturePointHistory<dim> "
                                 "instance has not been initialized."));

  return (linear_hardening_modulus *
          (hardening_parameter +
           ((self_hardening) ? (1.0 - hardening_parameter) : 0.0)));
}



} // namespace gCP



#endif /* INCLUDE_POINT_HISTORY_H_ */
