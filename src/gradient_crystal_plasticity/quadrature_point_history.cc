
#include <gCP/quadrature_point_history.h>



namespace gCP
{



template <int dim>
QuadraturePointHistory<dim>::QuadraturePointHistory()
:
flag_init_was_called(false)
{}



template <int dim>
void QuadraturePointHistory<dim>::init(
  const RunTimeParameters::ScalarMicroscopicStressLawParameters
    &parameters,
  const unsigned int n_slips)
{
  this->n_slips             = n_slips;

  initial_slip_resistance   = parameters.initial_slip_resistance;

  linear_hardening_modulus  = parameters.linear_hardening_modulus;

  hardening_parameter       = parameters.hardening_parameter;

  slip_resistances          = std::vector<double>(
                                n_slips,
                                0.0 /*initial_slip_resistance*/);

  flag_init_was_called      = true;
}


template <int dim>
void QuadraturePointHistory<dim>::update_values(
  const unsigned int                      q_point,
  const std::vector<std::vector<double>>  &slips,
  const std::vector<std::vector<double>>  &old_slips)
{
  for (unsigned int slip_id_alpha = 0;
        slip_id_alpha < n_slips;
        ++slip_id_alpha)
    for (unsigned int slip_id_beta = 0;
          slip_id_beta < n_slips;
          ++slip_id_beta)
      slip_resistances[slip_id_alpha] +=
        get_hardening_matrix_entry(slip_id_alpha == slip_id_beta) *
        std::abs(slips[slip_id_beta][q_point] -
                  old_slips[slip_id_beta][q_point]);
}



} // namespace gCP

template gCP::QuadraturePointHistory<2>::QuadraturePointHistory();
template gCP::QuadraturePointHistory<3>::QuadraturePointHistory();

template void gCP::QuadraturePointHistory<2>::init(
  const RunTimeParameters::ScalarMicroscopicStressLawParameters &,
  const unsigned int                                            );
template void gCP::QuadraturePointHistory<3>::init(
  const RunTimeParameters::ScalarMicroscopicStressLawParameters &,
  const unsigned int                                            );

template void gCP::QuadraturePointHistory<2>::update_values(
  const unsigned int,
  const std::vector<std::vector<double>>  &,
  const std::vector<std::vector<double>>  &);
template void gCP::QuadraturePointHistory<3>::update_values(
  const unsigned int,
  const std::vector<std::vector<double>>  &,
  const std::vector<std::vector<double>>  &);
