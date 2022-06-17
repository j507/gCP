#ifndef INCLUDE_POINT_HISTORY_H_
#define INCLUDE_POINT_HISTORY_H_

#include <gCP/run_time_parameters.h>



namespace gCP
{



template <int dim>
struct QuadraturePointHistory
{
public:
  /*!
   * @brief Default constructor
   *
   */
  QuadraturePointHistory();

  /*!
   * @brief Default destructor
   *
   */
  virtual ~QuadraturePointHistory() = default;

  /*!
   * @brief Get the slip resistance value at the quadrature point
   * of a slip system
   *
   * @param slip_id The id number of the slip system
   * @return double The slip resistance value corresponding to
   * @ref slip_id
   */
  double get_slip_resistance(const unsigned int slip_id) const;

  /*!
   * @brief Get the slip resistance value at the quadrature point
   * of all slip systems
   *
   * @return std::vector<double> The slip resistance value of all
   * slip systems
   */
  std::vector<double> get_slip_resistances() const;

  /*!
   * @brief Initiates the @ref QuadraturePointHistory instance
   *
   * @param parameters The material parameters of the evolution
   * equation
   * @param n_slips The number of slip systems of the crystal
   */
  void init(
    const RunTimeParameters::ScalarMicroscopicStressLawParameters
      &parameters,
    const unsigned int n_slips);

  /*!
   * @brief Stores the values of @ref slip_resistances in @ref
   * tmp_slip_resistances
   *
   * @details The values are stored in order to reset
   * @ref slip_resistances in @ref update_values
   */
  void store_current_values();

  /*!
   * @brief Updates the slip resistance values at the quadratue point
   *
   * @details The slip resitance values at the quadrature point @ref
   * q_point are computed using the temporally discretized evolution
   * equation
   *
   * \f[
   *    g^{n}_{\alpha} =
   *     g^{n-1}_{\alpha} +
   *     \sum_1^{\text{n_slips}} h_{\alpha\beta}
   *      \abs{\gamma^{n}_{\alpha} - \gamma^{n-1}_{\alpha}}
   * \f]
   *
   * @param q_point The quadrature point at which the slip resitance
   * values are updated
   * @param slips The slip values at t^{n}
   * @param old_slips The slip values at t^{n-1}
   * @todo Docu
   */
  void update_values(
    const unsigned int                      q_point,
    const std::vector<std::vector<double>>  &slips,
    const std::vector<std::vector<double>>  &old_slips);

private:
  unsigned int        n_slips;

  double              initial_slip_resistance;

  std::vector<double> slip_resistances;

  std::vector<double> tmp_slip_resistances;

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
